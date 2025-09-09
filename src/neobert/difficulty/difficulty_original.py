# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import logging
import datetime
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)#not sure why
 
from transformers.tokenization_utils_base import BatchEncoding


import os

from ..model import NeoBERTConfig, NeoBERTLMHead
from ..tokenizer import get_tokenizer
from ..dataset import get_dataset
from ..dataloader import get_dataloader
from ..scheduler import get_scheduler
from ..optimizer import get_optimizer

from omegaconf import OmegaConf
import torch
import torch.nn as nn
import pandas as pd

from tqdm import tqdm
from typing import Optional

from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from accelerate.utils import DistributedDataParallelKwargs

from peft import LoraConfig, get_peft_model, TaskType

from datasets import load_dataset,load_from_disk, Dataset,DatasetDict
from omegaconf import OmegaConf, DictConfig
import os

from torch.utils.data import DataLoader

# pretrained_SST2 = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
# pretrained_WNLI = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-WNLI")
# pretrained_STS_B = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-STS-B")
# pretrained_RTE = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-RTE")
# pretrained_QQP = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-QQP")
# pretrained_QNLI = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-QNLI")
# pretrained_MRPC = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-MRPC")
# pretrained_MNLI = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-MNLI")
# pretrained_COLA = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-CoLA")

class STSBDataCollator:
    """
    Collates batches for STSB regression task from already-tokenized datasets.
    Converts attention_mask to additive mask for NeoBERT.
    """
    def __init__(self, dtype_pad_mask):
        self.dtype_pad_mask = dtype_pad_mask

    def __call__(self, batch):
        # batch: list of dicts with tokenized fields
        input_ids = torch.stack([torch.tensor(ex["input_ids"]) for ex in batch])
        # Standard attention mask: 1 for tokens, 0 for padding
        attention_mask = torch.stack([torch.tensor(ex["attention_mask"]) for ex in batch])
        additive_mask = torch.where(attention_mask == 1, 0.0, float('-inf')).to(dtype = self.dtype_pad_mask)
        #additive_mask = torch.where(attention_mask == 1, 0.0, -1e4).to(dtype = self.dtype_pad_mask)

        labels = torch.tensor([float(ex["label"]) for ex in batch], dtype=torch.float32)
        collated = {
            "input_ids": input_ids,
            "attention_mask": additive_mask,
            "labels": labels
        }
        # If token_type_ids exist (for BERT), add them
        if "token_type_ids" in batch[0]:
            collated["token_type_ids"] = torch.stack([torch.tensor(ex["token_type_ids"]) for ex in batch])
        return collated

def to_target_batch_size(
    batch: BatchEncoding,
    stored_batch: BatchEncoding,
    target_size: int = 8,
):
    tmp = {}
    batch_size = batch["input_ids"].shape[0]

    # If the batch is toO large, we store samples
    if batch_size > target_size:
        for key in batch.keys():
            tmp[key] = torch.split(batch[key], [target_size, batch_size - target_size], dim=0)
            batch[key] = tmp[key][0]
            stored_batch[key] = tmp[key][1] if stored_batch[key] is None else torch.cat([tmp[key][1], stored_batch[key]], dim=0)

    # If the batch is to small, we fetch stored samples
    elif batch_size < target_size and stored_batch["input_ids"] is not None:
        stored_batch_size = stored_batch["input_ids"].shape[0]
        missing = target_size - batch_size

        # Fetch only necessary samples if storage is larger than required
        if missing < stored_batch_size:
            for key in batch.keys():
                stored_batch[key].to(batch[key].device)
                tmp[key] = torch.split(stored_batch[key], [missing, stored_batch_size - missing], dim=0)
                batch[key] = torch.cat([batch[key], tmp[key][0]], dim=0)
                stored_batch[key] = tmp[key][1]
                stored_batch[key].to("cpu", non_blocking=True)

        # Concatenate otherwise
        else:
            for key in batch.keys():
                batch[key] = torch.cat([batch[key], stored_batch[key]], dim=0)
                stored_batch[key] = None

    return batch, stored_batch

def measure_difficulty(cfg_dif):

    #SET UP ACCELERATOR---------------------------
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,  # enable manual control of the scheduler
        mixed_precision=cfg_dif.trainer.mixed_precision,
        gradient_accumulation_steps=cfg_dif.trainer.gradient_accumulation_steps,
        log_with="wandb",
        #project_config=project_config,print
        kwargs_handlers=[kwargs],
    )

    # Initialise the wandb run and pass wandb parameters
    os.makedirs(cfg_dif.wandb.dir, exist_ok=True)
    accelerator.init_trackers(
        project_name=cfg_dif.wandb.project,
        init_kwargs={
            "wandb": {
                "name": cfg_dif.wandb.name,
                "entity": cfg_dif.wandb.entity,
                "config": OmegaConf.to_container(cfg_dif) | {"distributed_type": accelerator.distributed_type},
                "tags": cfg_dif.wandb.tags,
                "dir": cfg_dif.wandb.dir,
                "mode": cfg_dif.wandb.mode,
                "resume": cfg_dif.wandb.resume,
            }
        },
    )

    set_seed(cfg_dif.seed)

    # Enable TF32 on matmul and on cuDNN
    torch.backends.cuda.matmul.allow_tf32 = cfg_dif.trainer.tf32
    torch.backends.cudnn.allow_tf32 = cfg_dif.trainer.tf32

     # Get the dtype for the pad_mask
    dtype_pad_mask = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     dtype_pad_mask = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     dtype_pad_mask = torch.bfloat16


    #END SET UP ACCELERATOR---------------------------

    #load my pretrained neobert  model
    cfg_path = os.path.join(cfg_dif.saved_model.base_path,"config.yaml")
    cfg = OmegaConf.load(cfg_path)

    tokenizer = get_tokenizer(**cfg.tokenizer) #should be the exact same as the  one used for the pretrained models
    neobert_model = NeoBERTLMHead(NeoBERTConfig(**cfg.model, **cfg.tokenizer, pad_token_id=tokenizer.pad_token_id))
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

    # get pretrained finetuned bert model
    

    # 2. Load pretrained STS-B model
    difmeasure_model_name = "textattack/bert-base-uncased-STS-B"
    #we check that the two tokenizers have same vocab size
    tokenizer_text_attack = AutoTokenizer.from_pretrained(difmeasure_model_name)
    assert tokenizer_text_attack.vocab_size == tokenizer.vocab_size, "The tokenizer vocab size does not match the pretrained model's tokenizer vocab_size."

    difmeasure_model = AutoModelForSequenceClassification.from_pretrained(difmeasure_model_name)
    difmeasure_model.eval()  # evaluation mode


    #load and cache datasets

    def prepare_and_cache_stsb(tokenizer, cache_dir: Optional[str] = None):
        """
        Download, tokenize, and cache the GLUE STSB dataset with only necessary columns.
        Only tokenized columns and label are kept.
        """
        stsb = load_dataset("glue", "stsb")  # DatasetDict with splits

        def tokenize_fn(batch):
            # Tokenize both sentences
            tokens = tokenizer(
                batch["sentence1"],
                batch["sentence2"],
                truncation=True,
                max_length=128,
                padding="max_length",
            )
            # Only add label once
            tokens["label"] = batch["label"]
            return tokens

        # Remove sentence1 and sentence2 from the output
        columns_to_remove = ["sentence1", "sentence2"]
        # Also remove any other columns except label and idx if you want
        for split in stsb.keys():
            # Find all columns except label and idx
            all_columns = stsb[split].column_names
            remove_cols = [col for col in all_columns if col not in ["label", "idx"]]
            stsb[split] = stsb[split].map(tokenize_fn, batched=True, remove_columns=remove_cols)

        # Choose cache dir
        if cache_dir is None:
            base_dir = Path("/data") if Path("/data").exists() else Path.home()
            cache_dir = base_dir / ".pathways_cache" / "glue_stsb_tokenized"
        else:
            cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Save each split
        for split in stsb.keys():
            stsb[split].save_to_disk(cache_dir / split)
        logger.info(f"Tokenized STSB saved to {cache_dir}")

    def load_cached_stsb(tokenizer, cache_dir: Optional[str] = None) -> dict:
        """
        Load cached tokenized STSB splits from disk.
        If not all splits are found, prepare and cache them first.
        Returns a dict: {"train": Dataset, "validation": Dataset, "test": Dataset}
        """
        if cache_dir is None:
            base_dir = Path("/data") if Path("/data").exists() else Path.home()
            cache_dir = base_dir / ".pathways_cache" / "glue_stsb_tokenized"
        else:
            cache_dir = Path(cache_dir)
        splits = ["train", "validation", "test"]
        datasets = {}
        missing = []
        for split in splits:
            split_path = cache_dir / split
            if split_path.exists():
                datasets[split] = load_from_disk(split_path)
            else:
                missing.append(split)
        if missing:
            logger.info(f"Missing splits {missing} in cache. Preparing and caching STSB dataset.")
            prepare_and_cache_stsb(tokenizer, cache_dir=cache_dir)
            for split in missing:
                split_path = cache_dir / split
                datasets[split] = load_from_disk(split_path)
        return datasets
    cached_dir = Path("/data/.pathways_cache/glue_stsb_tokenized")
    splits = load_cached_stsb(tokenizer, cache_dir=cached_dir)

    stsb_train_dataset = splits["train"]
    stsb_validation_dataset = splits["validation"]
    #stsb_test_dataset = splits["test"]


    # stsb_train_dataset = get_dataset(cfg #for the tokenizer, 
    #         **cfg_dif.dataset.train)
    # stst_validation_dataset = get_dataset(cfg, 
    #         **cfg_dif.dataset.validation)#need to change the config accordingly
    # #in the config I need to have subset made very precise that i want "stsb"
    
    # stsb = load_dataset("glue", "stsb")

    # # 1. Load STS-B validation set
    # stsb = load_dataset("glue", "stsb")
    # val_set = stsb["validation"]

    
    # 3. Loop over validation set and compute predictions + squared error (batched)


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

    stsb_collator = STSBDataCollator(dtype_pad_mask=dtype_pad_mask)

    #build dataloaders
    train_dataloader = DataLoader(
        stsb_train_dataset,
        batch_size=cfg_dif.dataloader.train.batch_size,
        shuffle=True,
        collate_fn=stsb_collator
    )
    val_dataloader = DataLoader(
        stsb_validation_dataset,
        batch_size=cfg_dif.dataloader.test.batch_size,
        shuffle=False,
        collate_fn=stsb_collator
    )


    #define optimizer

        # Optimizer and Scheduler
    if cfg_dif.scheduler.warmup_steps == None:
        warmup_steps = int(cfg_dif.scheduler.warmup_percent*cfg_dif.trainer.max_steps)
    else:
        warmup_steps = cfg_dif.scheduler.warmup_steps
    if cfg_dif.scheduler.decay_steps == None:
        decay_steps = int(cfg_dif.scheduler.decay_percent * cfg_dif.trainer.max_steps)
    else:
        decay_steps = cfg_dif.scheduler.decay_steps

    optimizer = get_optimizer(neobert_stsb_model, accelerator.distributed_type, name=cfg_dif.optimizer.name, **cfg_dif.optimizer.hparams)
    scheduler = get_scheduler(optimizer=optimizer, lr=cfg_dif.optimizer.hparams.lr,warmup_steps=warmup_steps, decay_steps=decay_steps, decay=cfg_dif.scheduler.decay)

    lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["qkv", "wo","in_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,  # <— avoids labels
    )


    # 2. Wrap your predictor model with LoRA
    for name, param in neobert_stsb_model.neobert_backbone.backbone.named_parameters():
        if "qkv" in name or "wo" in name or "in_proj" in name or "out_proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    #neobert_stsb_model.neobert_backbone = get_peft_model(neobert_stsb_model.neobert_backbone, lora_config)

    # Prepare with accelerate
    train_dataloader,val_dataloader,neobert_stsb_model, difmeasure_model, optimizer, scheduler = accelerator.prepare(
        train_dataloader,
        val_dataloader,
        neobert_stsb_model,
        difmeasure_model,
        optimizer,
        scheduler,
    )

    #neobert_model = torch.compile(neobert_model)

    #POur l'instant on ne s'en préoccupe pas
    # # Resume from the latest checkpoint
    # skipped_train_dataloader = None
    # if cfg.trainer.resume and os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
    #     accelerator.load_state()
    #     train_dataloader.set_epoch(metrics["train/epochs"])
    #     skipped_train_dataloader = accelerator.skip_first_batches(train_dataloader, metrics["train/batches"] % len(train_dataloader))

    # Progress bar
    step = 0
    pbar = tqdm(
        desc="Train",
        unit="step",
        initial=step,
        total=cfg_dif.trainer.max_steps,
        disable=(cfg_dif.trainer.disable_tqdm or not accelerator.is_main_process),
    )
    compiled_neobert_stsb_model = torch.compile(neobert_stsb_model)
    # while cfg_dif.trainer.max_steps > step:
        
    #     stored_batch = {
    #         "input_ids": None,
    #         "attention_mask": None,
    #         "labels": None,
    #     }
    #     for batch in train_dataloader:
    #         if batch["input_ids"].shape[0] != cfg_dif.dataloader.train.batch_size:
    #             batch, stored_batch = to_target_batch_size(batch, stored_batch, cfg_dif.dataloader.train.batch_size)
    #         if batch["input_ids"].shape[0] < cfg_dif.dataloader.train.batch_size:
    #             stored_batch = batch

    #             continue
            
    #         # Forward pass
    #         outputs = compiled_neobert_stsb_model(
    #             input_ids=batch["input_ids"],
    #             attention_mask=batch["attention_mask"],
    #             labels=batch["labels"]
    #         )
    #         loss = outputs["loss"]  # This is already MSE loss from your model's forward()
    #         expert_loss = outputs["expert_usage_loss"]
    #         # print(f"Loss: {loss}, Expert Loss: {expert_loss}")
    #         #now print their dtypes
    #         # print(f"Loss dtype: {loss.dtype}, Expert Loss dtype: {expert_loss.dtype}")

    #         # Under the no_sync context manager, PyTorch will skip synchronizing the gradients when .backward() is
    #         # called, and the first call to .backward() outside this context manager will trigger the synchronization.
    #         # Accumulating manually gives more flexibility and is compatible with TPUs.
    #         # if metrics["train/batches"] % cfg.trainer.gradient_accumulation_steps != 0:
    #         #     with accelerator.no_sync(BERTpredictor):
    #         #         loss = loss_fnt(predicted_routed_logits, target_expert_mask*1.0)
    #         #         loss.backward()
            
            
    #         accelerator.backward(loss)

    #         #accelerator.unscale_gradients(optimizer)

    #         num_regression_params = 0
    #         num_attention_params = 0
    #         num_expert_params = 0

    #         grad_norm_regression = 0
    #         grad_norm_attention = 0
    #         grad_norm_expert = 0

    #         for p in compiled_neobert_stsb_model.neobert_backbone.backbone.named_parameters():
    #             name, p = p
    #             if p.requires_grad:
    #                 if "qkv" in name or "wo" in name:
    #                     num_attention_params += p.numel()
    #                     if p.grad is not None:
    #                         grad_norm_attention += p.grad.data.norm(2).item() ** 2
    #                 if "in_proj" in name or "out_proj" in name:
    #                     num_expert_params += p.numel()
    #                     if p.grad is not None:
    #                         grad_norm_expert += p.grad.data.norm(2).item() ** 2

    #         for p in compiled_neobert_stsb_model.regression_head.parameters():
    #             if p.grad is not None:
    #                 param_norm = p.grad.data.norm(2)
    #                 grad_norm_regression += param_norm.item() ** 2
    #                 num_regression_params += p.numel()
            
    #         grad_norm_attention = grad_norm_attention ** 0.5
    #         grad_norm_expert = grad_norm_expert ** 0.5
    #         grad_norm_regression = grad_norm_regression ** 0.5

    #         accelerator.log({
    #             "train/regression_grad_norm": grad_norm_regression,
    #             "train/attention_grad_norm": grad_norm_attention,
    #             "train/expert_grad_norm": grad_norm_expert,
    #             "step": step
    #         })

    #         relative_grad_norm_attention = grad_norm_attention/num_attention_params
    #         relative_grad_norm_expert = grad_norm_expert/num_expert_params 
    #         relative_grad_norm_regression = grad_norm_regression/num_regression_params 

    #         accelerator.log({
    #             "train/rel_attention_grad_norm": relative_grad_norm_attention,
    #             "train/rel_expert_grad_norm": relative_grad_norm_expert,
    #             "train/rel_regression_grad_norm": relative_grad_norm_regression,
    #             "step": step
    #         })

    #         # # Log gradient norm of regression head before clipping
    #         # regression_head = compiled_neobert_stsb_model.regression_head
    #         # grad_norm = None
    #         # total_norm = 0.0
    #         # # total_grad_tensor = None  # Added to accumulate total gradient tensor
    #         # for p in regression_head.parameters():
    #         #     if p.grad is not None:
    #         #         param_norm = p.grad.data.norm(2)
    #         #         total_norm += param_norm.item() ** 2
    #         #         num_regression_params += p.numel()
    #         #         # Accumulate total gradient tensor
    #         #         if total_grad_tensor is None:
    #         #             total_grad_tensor = p.grad.data.clone().flatten()
    #         #         else:
    #         #             total_grad_tensor = torch.cat([total_grad_tensor, p.grad.data.flatten()])
    #         # grad_norm = total_norm ** 0.5
    #         # relative_grad_norm = grad_norm/num_regression_params
    #         # accelerator.log({"train/regression_head_grad_norm": grad_norm, "step": step})
    #         # accelerator.log({"train/rel_regression_head_grad_norm": relative_grad_norm, "step": step})
    #         # # Print the total gradient tensor for the regression head
    #         # print("Total regression head gradient tensor:", total_grad_tensor)

    #         # Log embedding grad norm for comparison
    #         # embedding_head = compiled_neobert_stsb_model.neobert_backbone.backbone.encoder
    #         # grad_norm = None
    #         # total_norm = 0.0
    #         # for p in embedding_head.parameters():
    #         #     if p.grad is not None:
    #         #         param_norm = p.grad.data.norm(2)
    #         #         total_norm += param_norm.item() ** 2
    #         #         num_embedding_params += p.numel()
    #         # grad_norm = total_norm ** 0.5
    #         # relative_grad_norm = grad_norm/num_embedding_params
    #         # accelerator.log({"train/embedding_head_grad_norm": grad_norm, "step": step})
    #         # accelerator.log({"train/rel_embedding_head_grad_norm": relative_grad_norm, "step": step})

            


    #         if cfg_dif.trainer.gradient_clipping is not None and cfg_dif.trainer.gradient_clipping > 0:
    #             accelerator.clip_grad_norm_(compiled_neobert_stsb_model.parameters(), cfg_dif.trainer.gradient_clipping)

    #         # Update model parameters
    #         optimizer.step()
    #         scheduler.step()
    #         optimizer.zero_grad()

    #         step += 1
    #         pbar.update(1)

    #         if step % cfg_dif.wandb.log_interval == 0:
    #             accelerator.log({"train/loss": loss.item(), "step": step})
    #         if step >= cfg_dif.trainer.max_steps:
    #                 break
        
    # pbar.close()

    # # --- Evaluate on validation set and build df and expert_usage_df together ---

    # Set both models to eval mode
    neobert_stsb_model.eval()
    difmeasure_model.eval()

    difficulty_estimates = []
    expert_usage_outputs = []

    

    with torch.no_grad():
        for batch in val_dataloader:
            # --- BERT difficulty ---
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            idxs = batch.get("idx", None)

            outputs_bert = difmeasure_model(
                input_ids=input_ids,
                attention_mask=(attention_mask == 0).long() if attention_mask.dtype != torch.long else attention_mask
            )
            pred_scores = outputs_bert.logits.squeeze(-1).cpu().numpy()
            gold_scores = labels.cpu().numpy()
            mse_errors = (pred_scores - gold_scores) ** 2

            input_ids_list = input_ids.cpu().tolist()
            for i in range(len(gold_scores)):
                result = {
                    "gold": gold_scores[i],
                    "pred": pred_scores[i],
                    "mse_error": mse_errors[i],
                    "input_ids": input_ids_list[i],  # add input_ids to difficulty_estimates
                }
                if idxs is not None:
                    result["idx"] = idxs[i].item() if hasattr(idxs[i], "item") else idxs[i]
                difficulty_estimates.append(result)
                

            # --- NeoBERT expert usage ---
            outputs_neobert = neobert_stsb_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels if "labels" in batch else None
            )
            expert_usage_loss_batch = outputs_neobert["expert_usage_loss"].cpu().tolist()
            idx_batch = batch.get("idx", None)
            if idx_batch is not None:
                idx_batch = idx_batch.cpu().tolist()
                for idx, expert_loss in zip(idx_batch, expert_usage_loss_batch):
                    expert_usage_outputs.append({
                        "idx": idx,
                        "expert_usage_loss": expert_loss
                    })
            else:
                input_ids_batch = input_ids.cpu().tolist()
                for ids, expert_loss in zip(input_ids_batch, expert_usage_loss_batch):
                    expert_usage_outputs.append({
                        "input_ids": ids,
                        "expert_usage_loss": expert_loss
                    })

    difficulty_estimates_df = pd.DataFrame(difficulty_estimates)
    print(difficulty_estimates_df.head())

    expert_usage_df = pd.DataFrame(expert_usage_outputs)
    print(expert_usage_df.head())

    print(f"Number of rows in df (mse results): {len(difficulty_estimates_df)}")
    print(f"Number of rows in expert_usage_df: {len(expert_usage_df)}")
    print(f"Same length? {len(difficulty_estimates_df) == len(expert_usage_df)}")

    # --- Merge DataFrames ---
    if "idx" in difficulty_estimates_df.columns and "idx" in expert_usage_df.columns:
        merged_df = pd.merge(difficulty_estimates_df, expert_usage_df, on="idx")
    else:
        # Convert input_ids to string for merging
        difficulty_estimates_df["input_ids_str"] = difficulty_estimates_df["input_ids"].apply(lambda x: str(x))
        expert_usage_df["input_ids_str"] = expert_usage_df["input_ids"].apply(lambda x: str(x))
        merged_df = pd.merge(difficulty_estimates_df, expert_usage_df, left_on="input_ids_str", right_on="input_ids_str")
        # Remove input_ids_str columns and keep only one input_ids column
        merged_df = merged_df.drop(columns=["input_ids_str"])
        # If there are two input_ids columns, drop one and rename the other to 'input_ids'
        if "input_ids_x" in merged_df.columns and "input_ids_y" in merged_df.columns:
            merged_df = merged_df.drop(columns=["input_ids_y"])
            merged_df = merged_df.rename(columns={"input_ids_x": "input_ids"})


    print(f"Number of rows in merged_df: {len(merged_df)}")
    print(merged_df.head())
    # Save merged_df to /runs/glue_stsb
    save_dir = Path("/runs/glue_stsb")
    save_dir.mkdir(parents=True, exist_ok=True)
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = cfg.model.type + f"_merged_df_{time_str}.csv"

    merged_df.to_csv(save_dir / filename, index=False)


