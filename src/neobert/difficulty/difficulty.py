import os
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
import numpy as np
#from scipy.stats import pearsonr not on modal we do the analysis locally
from ..model import NeoBERTConfig, NeoBERTLMHead
import datetime

from pathlib import Path

def measure_difficulty(cfg_dif):
    import os
    import pandas as pd
    from omegaconf import OmegaConf
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    
    # Load pretrained TextAttack model
    textattack_model_name = "textattack/bert-base-uncased-STS-B"
    textattack_tokenizer = AutoTokenizer.from_pretrained(textattack_model_name)
    textattack_model = AutoModelForSequenceClassification.from_pretrained(textattack_model_name)
    textattack_model.eval()
    
    # Load validation set
    stsb = load_dataset("glue", "stsb")
    val_set = stsb["validation"]

    class STSBDataset(Dataset):
        def __init__(self, val_set):
            self.sentence1 = val_set["sentence1"]
            self.sentence2 = val_set["sentence2"]
            self.labels = val_set["label"]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "sentence1": self.sentence1[idx],
                "sentence2": self.sentence2[idx],
                "label": self.labels[idx]
            }

    def collate_fn(batch):
        s1 = [item["sentence1"] for item in batch]
        s2 = [item["sentence2"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
        # Tokenize for both models (same tokenizer for consistency)
        inputs = textattack_tokenizer(
            s1,
            s2,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "sentence1": s1,
            "sentence2": s2,
            "labels": labels,
            "inputs": inputs
        }

    dataset = STSBDataset(val_set)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    cfg_path = os.path.join(cfg_dif.saved_model.base_path,"config.yaml")
    cfg = OmegaConf.load(cfg_path)

    #tokenizer = get_tokenizer(**cfg.tokenizer) #should be the exact same as the  one used for the pretrained models
    neobert_model = NeoBERTLMHead(NeoBERTConfig(**cfg.model, **cfg.tokenizer, pad_token_id=textattack_tokenizer.pad_token_id)) # à voir quel tokenizer on utilise. IL va falloir les regarder de près les deux
    state_dict_path = os.path.join(cfg_dif.saved_model.base_path, "model_checkpoints", cfg_dif.saved_model.checkpoint,"state_dict.pt")
    neobert_state_dict = torch.load(state_dict_path, map_location="cpu")

    # Fix keys: strip "_orig_mod." if present
    new_state_dict = {}
    for k, v in neobert_state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod."):]] = v
        else:
            new_state_dict[k] = v

    neobert_model.load_state_dict(new_state_dict, strict=True)
    neobert_backbone = neobert_model.model #ignore the LM head

    class NeoBERTBackboneWrapper(nn.Module):
        """
        Wraps NeoBERT backbone to accept HuggingFace-style arguments for LoRA compatibility.
        """
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, output_router_logits=True, output_expert_usage_loss=False, **kwargs):
            # Map input_ids -> src, attention_mask -> pad_mask
            src = input_ids
            pad_mask = attention_mask
            # Pass through to NeoBERT
            return self.backbone(src=src, pad_mask=pad_mask, output_router_logits=output_router_logits, output_expert_usage_loss=output_expert_usage_loss)


    class NeoBERTForSTSB(nn.Module):
        def __init__(self, hidden_size, dropout_prob=0.1):
            super().__init__()
            # Wrap backbone for LoRA compatibility
            self.neobert_backbone = NeoBERTBackboneWrapper(neobert_backbone)
            self.dropout = nn.Dropout(dropout_prob)
            self.regression_head = nn.Linear(hidden_size, 1)  # single float output
            self.config = neobert_backbone.config # not quite same as cfg since its a NeoBERTconfig but we could equivalently have used cfg
            self.expert_dims = [int(expert_size) for expert_size in self.config.expert_sizes.split(",")]

        def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
            backbone_outputs = self.neobert_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_expert_usage_loss=True
            )

            # print("input_ids:", input_ids)
            # print("attention_mask:", attention_mask)
            # print("labels:", labels)

            final_hidden_state = backbone_outputs["hidden_representation"]
            router_logits  = backbone_outputs["router_logits"]

            pooled_output = final_hidden_state[:, 0, :]
            pooled_output = self.dropout(pooled_output)
            # # print("pooled_output:", pooled_output)
            # print(self.regression_head)
            
            # Print all regression_head parameters
            # print("regression_head parameters:")
            # for name, param in self.regression_head.named_parameters():
            #     print(f"{name}: {param.data}")

        # # Print min and max of regression_head parameters
        #     params = list(self.regression_head.parameters())
        #     if params:
        #         weights = params[0]
        #         bias = params[1] if len(params) > 1 else None
        #         print("regression_head weights min:", weights.min().item(), "max:", weights.max().item())
        #         if bias is not None:
        #             print("regression_head bias min:", bias.min().item(), "max:", bias.max().item())

            logits = self.regression_head(pooled_output).squeeze(-1)
            # print("logits:", logits)

            loss = None
            if labels is not None:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.float(), labels.float())

            expert_usage_loss = self._MoEBlockRoutingCost(router_logits, attention_mask) #(batch_size,)

            return {"loss": loss, "logits": logits, "expert_usage_loss": expert_usage_loss}

        def _MoEBlockRoutingCost(self,
                            router_logits, #tuple of n_layers of tensors of size (batch * sequence_length, n_experts)
                            attention_mask #shape (batch,seq_length)
                            ):

            # Correct stacking of tuple of raw_routing_weights            
            router_logits = torch.stack(list(router_logits), dim=0) #shape (n_layers, batch * sequence_length, n_experts)
            raw_routing_weights = torch.softmax(router_logits, dim=-1)

            if len(self.expert_dims) > 1:
                expert_dims_tensor = torch.tensor(self.expert_dims,dtype=torch.float32,
                    device=raw_routing_weights.device )
                normalisation_factor = torch.sum(expert_dims_tensor)
                normalised_expert_sizes = expert_dims_tensor / normalisation_factor#prevent overflow
                routing_costs = normalised_expert_sizes**self.config.expert_cost_exponent
                routing_costs = routing_costs.to(dtype = raw_routing_weights.dtype)#cast back to original dtype

                n_layers,_,n_experts = raw_routing_weights.shape
                batch_size, seq_length = attention_mask.shape
                raw_routing_weights = raw_routing_weights.reshape(n_layers, batch_size, seq_length, n_experts)
                number_of_tokens_per_seq = seq_length
                

                #ignore padding_tokens
                if attention_mask!= None: #attention_mask has shape (Batch,seq_length) and routing_weights has shape (n_layers, batch * sequence_length, n_experts)
                    multiplicative_mask = torch.where(attention_mask == 0, 1.0, 0.0).to(dtype = raw_routing_weights.dtype).reshape(1, batch_size, seq_length, 1)
                    number_of_tokens_per_seq = multiplicative_mask.sum(dim=2).squeeze()
                    raw_routing_weights = raw_routing_weights * multiplicative_mask
                # Compute usage loss per token
                expert_usage_loss = torch.einsum('mijk,k->mij', raw_routing_weights, routing_costs) #(n_layers,batch_size,seq_length)
                
                #sum across layers
                expert_usage_loss = torch.sum(expert_usage_loss, dim=0) #shape (batch_size,seq_length)
                # Average over all tokens in the sequence
                normalised_mean_expert_usage_loss = torch.sum(expert_usage_loss, dim = 1) / number_of_tokens_per_seq #shape (batch_size,)
            else:
                normalised_mean_expert_usage_loss = None
                
            return normalised_mean_expert_usage_loss # the normalisation is not an issue as we care about comparisons but just a reminder it is here
        



    neobert_stsb_model = NeoBERTForSTSB(
        hidden_size=cfg.model.hidden_size,
        dropout_prob=0.1
    )
    neobert_stsb_model.eval()
    
    # Evaluation lists
    textattack_preds = []
    neobert_expert_usage_losses = []
    gold_labels = []
    all_sentence1 = []
    all_sentence2 = []

    print("Evaluating both models on validation set...")
    
    for batch_idx, batch in enumerate(dataloader):
        batch_inputs = batch["inputs"]
        batch_labels = batch["labels"]
        batch_sentence1 = batch["sentence1"]
        batch_sentence2 = batch["sentence2"]

        # TextAttack model evaluation
        with torch.no_grad():
            textattack_outputs = textattack_model(**batch_inputs)
            textattack_pred = textattack_outputs.logits.squeeze(-1).cpu().numpy()

        # NeoBERT model evaluation
        neobert_inputs = dict(batch_inputs)
        # NeoBERT expects float attention mask: 0.0 for tokens, -inf for padding
        neobert_inputs['attention_mask'] = torch.where(
            neobert_inputs['attention_mask'] == 1, 0.0, -torch.inf
        ).to(dtype=torch.float32)

        with torch.no_grad():
            neobert_outputs = neobert_stsb_model(
                input_ids=neobert_inputs['input_ids'],
                attention_mask=neobert_inputs['attention_mask'],
                labels=batch_labels
            )
            expert_usage_loss = neobert_outputs["expert_usage_loss"]
            # expert_usage_loss: shape (batch_size,)
            if expert_usage_loss is not None:
                expert_usage_loss = expert_usage_loss.cpu().numpy()
            else:
                expert_usage_loss = np.zeros(len(batch_labels))

        # Store results
        textattack_preds.extend(textattack_pred.tolist())
        neobert_expert_usage_losses.extend(expert_usage_loss.tolist())
        gold_labels.extend(batch_labels.cpu().numpy().tolist())
        all_sentence1.extend(batch_sentence1)
        all_sentence2.extend(batch_sentence2)

        if batch_idx % 10 == 0:
            print(f"Processed {batch_idx * 32 + len(batch_labels)} examples...")

    # Convert to numpy arrays
    textattack_preds = np.array(textattack_preds)
    gold_labels = np.array(gold_labels)
    neobert_expert_usage_losses = np.array(neobert_expert_usage_losses)
    
    # Calculate MSE for TextAttack model
    mse_errors = (textattack_preds - gold_labels) ** 2
    overall_mse = np.mean(mse_errors)
    
    # Create DataFrame
    results_df = pd.DataFrame({
        'sentence1': all_sentence1,
        'sentence2': all_sentence2,
        'gold_label': gold_labels,
        'textattack_pred': textattack_preds,
        'mse_error': mse_errors,
        'expert_usage_loss': neobert_expert_usage_losses
    })
    
    # Print summary statistics
    print(f"\nOverall Results:")
    print(f"TextAttack Model MSE: {overall_mse:.4f}")
    print(f"Mean Expert Usage Loss: {np.mean(neobert_expert_usage_losses):.4f}")
    print(f"Std Expert Usage Loss: {np.std(neobert_expert_usage_losses):.4f}")
    
    # Calculate correlation between MSE error and expert usage loss
    #if len(neobert_expert_usage_losses) > 0 and np.std(neobert_expert_usage_losses) > 0:
        #correlation = pearsonr(mse_errors, neobert_expert_usage_losses)[0]
        #print(f"Correlation between MSE error and Expert Usage Loss: {correlation:.4f}")
    
    print(results_df.head())
    # Save results_df to /runs/glue_stsb

    save_dir = Path("/runs/glue_stsb") #only works on Modal this one
    save_dir.mkdir(parents=True, exist_ok=True)
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = cfg.model.type + f"_results_df_{time_str}.csv"

    results_df.to_csv(save_dir / filename, index=False)
    
    return results_df