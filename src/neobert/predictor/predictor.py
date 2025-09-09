#from pyexpat import model

#from NeoBERT.NeoBERT_dev.src.neobert.pretraining import metrics
from ..tokenizer import get_tokenizer
from transformers import BertModel, BatchEncoding
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..dataloader import get_dataloader
from ..model import NeoBERTLMHead, NeoBERTConfig
from ..optimizer import get_optimizer
from ..scheduler import get_scheduler

from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from accelerate.utils import DistributedDataParallelKwargs

from peft import LoraConfig, get_peft_model, TaskType

from ..dataset import get_dataset
from omegaconf import OmegaConf, DictConfig
import os


#def config
#def wandb
#def appfunction depuis modal qu'on exécute
#def un fichier exécutable qui l'appelle elle et la config

#def imports

import wandb

# Initialize wandb once at the start

def to_target_batch_size(
    batch: BatchEncoding,
    stored_batch: BatchEncoding,
    target_size: int = 8,
):
    tmp = {}
    batch_size = batch["input_ids"].shape[0]

    # If the batch is to large, we store samples
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

#ajouter split_dataset_seed,test_size à la config, path to state dict
class SwiGLU(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim * 2, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, out_dim, bias=bias)

    def forward(self, x):
        x_proj = self.in_proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return self.out_proj(F.silu(x1) * x2)


class RouterPredictionHead(nn.Module):
        def __init__(self, num_experts, hidden_size, num_hidden_layers):
            super().__init__()
            self.mlp_block = nn.ModuleList(nn.Sequential(SwiGLU(hidden_size, hidden_size,hidden_size, bias=False),
                            nn.Linear(hidden_size, num_experts)) for _ in range(num_hidden_layers))

        def forward(self, x): #batch_dim, seq_length, hidden_size
            outputs = tuple(mlp(x) for mlp in self.mlp_block)## tuple of n_layers tensors of size batch_dim, seq_length, n_experts
            outputs = tuple(layer_output.view(-1, layer_output.size(-1)) for layer_output in outputs)
            return  outputs  # tuple of n_layers tensors of size batch_dim*seq_length, n_experts

class BERTPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        #pretrained bert params
        hidden_size = self.bert.config.hidden_size
        #neobert layer params
        num_hidden_layers = cfg.model.num_hidden_layers
        num_experts = len(cfg.model.expert_sizes.split(","))

        self.router_head = RouterPredictionHead(num_experts, hidden_size, num_hidden_layers)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels = None, *args, **kwargs):
        kwargs.pop("labels", None)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden_state = outputs.last_hidden_state
        logits = self.router_head(last_hidden_state)
        return logits
    

def get_expert_mask(gate_logits,cfg):
    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        stacked_gate_logits = torch.stack([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)#n_layers, batch_size*seq_length, n_experts
        _,_, num_experts = stacked_gate_logits.shape
    routing_weights = torch.nn.functional.softmax(stacked_gate_logits, dim=-1)


    if cfg.model.routing_strategy == "top_k":
        top_k = cfg.model.num_experts_per_tok_inference #if we wanted  to use top_k for heterogeneous moe
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        target_expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts).sum(2)#n_layers, batch_size*seq_length, n_experts
    
    elif cfg.model.routing_strategy == "top_p":
        top_p = cfg.model.min_expert_cumprob_per_token
        sorted_weights, sorted_indices = torch.sort(routing_weights, dim=-1, descending=False)

        cum_probs = sorted_weights.cumsum(dim=-1)
        mask = cum_probs > 1 - top_p

        unsorted_mask = torch.zeros_like(mask, dtype=torch.bool)
        target_expert_mask = unsorted_mask.scatter(dim=-1, index=sorted_indices, src=mask) #n_layers, batch_size*seq_length, n_experts

    return target_expert_mask # n_layers, batch_size*n_seq, n_experts


def predictor(cfg_predictor):


    #SET UP ACCELERATOR---------------------------

    # Accelerator object: I DONT CARE SO MUCH ABOUT LOGGING THE TRAINING
    # project_config = ProjectConfiguration(
    #     cfg_predictor.trainer.dir,

    #     automatic_checkpoint_naming=True,
    #     total_limit=cfg_predictor.trainer.accelerate.max_ckpt,
    #     iteration=iteration,
    # )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,  # enable manual control of the scheduler
        mixed_precision=cfg_predictor.trainer.mixed_precision,
        gradient_accumulation_steps=cfg_predictor.trainer.gradient_accumulation_steps,
        log_with="wandb",
        #project_config=project_config,
        kwargs_handlers=[kwargs],
    )

    # Initialise the wandb run and pass wandb parameters
    os.makedirs(cfg_predictor.wandb.dir, exist_ok=True)
    accelerator.init_trackers(
        project_name=cfg_predictor.wandb.project,
        init_kwargs={
            "wandb": {
                "name": cfg_predictor.wandb.name,
                "entity": cfg_predictor.wandb.entity,
                "config": OmegaConf.to_container(cfg_predictor) | {"distributed_type": accelerator.distributed_type},
                "tags": cfg_predictor.wandb.tags,
                "dir": cfg_predictor.wandb.dir,
                "mode": cfg_predictor.wandb.mode,
                "resume": cfg_predictor.wandb.resume,
            }
        },
    )

    set_seed(cfg_predictor.seed)

    # Enable TF32 on matmul and on cuDNN
    torch.backends.cuda.matmul.allow_tf32 = cfg_predictor.trainer.tf32
    torch.backends.cudnn.allow_tf32 = cfg_predictor.trainer.tf32

     # Get the dtype for the pad_mask
    dtype_pad_mask = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype_pad_mask = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype_pad_mask = torch.bfloat16

    #END SET UP ACCELERATOR---------------------------


    # device=torch.device("cuda" if torch.cuda.is_available() else "cpu") # en ai-je encore besoin?

    # wandb.init(name=cfg_predictor.wandb.name,
    #             project=cfg_predictor.wandb.project,
    #               entity = cfg_predictor.wandb.entity, 
    #               config=OmegaConf.to_container(cfg_predictor, resolve=True))
    
    # et de ca aussi?

    #load  pretrained config
    cfg_path = os.path.join(cfg_predictor.saved_model.base_path,"config.yaml")
    cfg = OmegaConf.load(cfg_path)

    # get pretrained BERT predictor
    BERTpredictor = BERTPredictor(cfg)
    #BERTpredictor.to(device)

    # get pretrained neobert model
    tokenizer = get_tokenizer(**cfg.tokenizer)
    neobert_model = NeoBERTLMHead(NeoBERTConfig(**cfg.model, **cfg.tokenizer, pad_token_id=tokenizer.pad_token_id))
    state_dict_path = os.path.join(cfg_predictor.saved_model.base_path, "model_checkpoints", cfg_predictor.saved_model.checkpoint,"state_dict.pt")
    neobert_state_dict = torch.load(state_dict_path, map_location="cpu")

    # Fix keys: strip "_orig_mod." if present
    new_state_dict = {}
    for k, v in neobert_state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod."):]] = v
        else:
            new_state_dict[k] = v

    neobert_model.load_state_dict(new_state_dict, strict=True)

    #freeze params
    neobert_model.eval()
    for params in neobert_model.parameters():
        params.requires_grad = False
    #neobert_model.to(device) made unnecessary by accelerator.prepare()

    #get dataset
    train_dataset = get_dataset(cfg, 
            **cfg_predictor.dataset.train)
    test_dataset = get_dataset(cfg, 
            **cfg_predictor.dataset.test)

    #build dataloaders
    train_dataloader = get_dataloader(train_dataset, tokenizer, dtype=dtype_pad_mask, **cfg_predictor.dataloader.train, **cfg_predictor.datacollator)
    test_dataloader = get_dataloader(test_dataset, tokenizer, dtype=dtype_pad_mask, **cfg_predictor.dataloader.test, **cfg_predictor.datacollator)

    # define loss function
    loss_fn = nn.BCEWithLogitsLoss()

    #define optimizer

     # Optimizer and Scheduler
    optimizer = get_optimizer(BERTpredictor, accelerator.distributed_type, name=cfg_predictor.optimizer.name, **cfg_predictor.optimizer.hparams)
    scheduler = get_scheduler(optimizer=optimizer, lr=cfg_predictor.optimizer.hparams.lr, **cfg_predictor.scheduler)
    #SANS DOUTE FAUT-IL UN SCHEDULER DIFFERENT ICI


    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,  # <— avoids labels
    )

    # 2. Wrap your predictor model with LoRA
    BERTpredictor.bert = get_peft_model(BERTpredictor.bert, lora_config)

    # Prepare with accelerate
    train_dataloader,test_dataloader, BERTpredictor,neobert_model, optimizer, scheduler = accelerator.prepare(
        train_dataloader,
        test_dataloader,
        BERTpredictor,
        neobert_model,
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
        total=cfg_predictor.trainer.max_steps,
        disable=(cfg_predictor.trainer.disable_tqdm or not accelerator.is_main_process),
    )

    #finetuning on layer prediction

    val_batches_per_eval = 50
    val_interval = 30

    compiled_BERTpredictor = torch.compile(BERTpredictor)
    while cfg_predictor.trainer.max_steps > step:
        stored_batch = {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
        }
        for batch in train_dataloader:
            #batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)} made useless by line264 which puts everything on the right device
            # Pack or truncate the batch to target batch size (batch size might be variable due to sequence packing).
            if batch["input_ids"].shape[0] != cfg_predictor.dataloader.train.batch_size:
                batch, stored_batch = to_target_batch_size(batch, stored_batch, cfg_predictor.dataloader.train.batch_size)

            # If it is still smaller, stored batches were not enough and we skip to the next iteration to fill the batch
            if batch["input_ids"].shape[0] < cfg_predictor.dataloader.train.batch_size:
                stored_batch = batch
                continue

            #compute the target routing paths
            neobert_model_output = neobert_model(batch["input_ids"], batch.get("attention_mask", None), output_expert_usage_loss=False, output_router_logits=True)
            target_gate_logits = neobert_model_output['router_logits'] #tuple  of n_layers of batch_size*seq_length,  n_experts

            target_expert_mask = get_expert_mask(target_gate_logits, cfg) # n_layers, batch_size*n_seq, n_experts

            predicted_routed_logits = compiled_BERTpredictor(input_ids=batch["input_ids"])
            predicted_routed_logits = torch.stack(predicted_routed_logits, dim=0) #n_layers, batch_size*seq_length, n_experts

            #ignore padded tokens
            pad_mask = batch.get("attention_mask", None)
            if pad_mask is not None:
                pad_mask = pad_mask.view(-1, 1)
                pad_mask = (pad_mask != float("-inf")).squeeze(-1)
                predicted_routed_logits = predicted_routed_logits[:,pad_mask,:]
                target_expert_mask = target_expert_mask[:,pad_mask,:]

            #reshape 
            predicted_routed_logits = predicted_routed_logits.view(-1, predicted_routed_logits.size(-1)) #n_layers*batch_size*seq_length, n_experts
            target_expert_mask = target_expert_mask.view(-1, target_expert_mask.size(-1)) #n_layers*batch_size*seq_length, n_experts

            # Under the no_sync context manager, PyTorch will skip synchronizing the gradients when .backward() is
            # called, and the first call to .backward() outside this context manager will trigger the synchronization.
            # Accumulating manually gives more flexibility and is compatible with TPUs.
            # if metrics["train/batches"] % cfg.trainer.gradient_accumulation_steps != 0:
            #     with accelerator.no_sync(BERTpredictor):
            #         loss = loss_fn(predicted_routed_logits, target_expert_mask*1.0)
            #         loss.backward()
            
            loss = loss_fn(predicted_routed_logits, target_expert_mask*1.0)
            accelerator.backward(loss)

            if cfg_predictor.trainer.gradient_clipping is not None and cfg_predictor.trainer.gradient_clipping > 0:
                accelerator.clip_grad_norm_(compiled_BERTpredictor.parameters(), cfg_predictor.trainer.gradient_clipping)

            # Update model parameters
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step += 1
            pbar.update(1)
    
            if step % cfg_predictor.wandb.log_interval == 0:
                accelerator.log({"train/loss": loss.item(), "step": step})
            if step >= cfg_predictor.trainer.max_steps:
                    break
            # Add validation/testing every val_interval steps
            if step % val_interval == 0:
                run_test_batches(BERTpredictor, neobert_model, test_dataloader, accelerator, cfg, max_batches=val_batches_per_eval)
    pbar.close()
  
#testing
    BERTpredictor.eval()
    with torch.no_grad():
        pbar_test = tqdm(
            test_dataloader,
            desc="Eval",
            unit="batch",
            disable=(cfg_predictor.trainer.disable_tqdm or not accelerator.is_main_process),
        )
        stored_batch = {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
        }
        total_acc = []
        for batch in test_dataloader:
            
            #batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)} made useless by accelerate.prepare which puts everything on the right device
    
            if batch["input_ids"].shape[0] != cfg_predictor.dataloader.train.batch_size:
                batch, stored_batch = to_target_batch_size(batch, stored_batch, cfg_predictor.dataloader.train.batch_size)

            # If it is still smaller, stored batches were not enough and we skip to the next iteration to fill the batch
            if batch["input_ids"].shape[0] < cfg_predictor.dataloader.train.batch_size:
                stored_batch = batch
                continue
            
            #target
            model_output = neobert_model(batch["input_ids"], batch.get("attention_mask", None), output_expert_usage_loss=False, output_router_logits=True)
            target_gate_logits = model_output['router_logits'] #tuple  of n_layers of batch_size*seq_length,  n_experts
            target_expert_mask = get_expert_mask(target_gate_logits, cfg) # n_layers, batch_size*n_seq, n_experts

            #prediction
            predicted_routed_logits = BERTpredictor(batch["input_ids"]) 
            predicted_expert_mask = get_expert_mask(predicted_routed_logits, cfg)#n_layers, batch_size*seq_length, n_experts

            #ignore padded tokens
            pad_mask = batch.get("attention_mask", None)
            if pad_mask is not None:
                pad_mask = pad_mask.view(-1, 1)
                pad_mask = (pad_mask != float("-inf")).squeeze(-1)
                predicted_expert_mask = predicted_expert_mask[:,pad_mask,:]
                target_expert_mask = target_expert_mask[:,pad_mask,:]

            #reshape
            target_expert_mask = target_expert_mask.view(-1, target_expert_mask.size(-1)) #n_layers*batch_size*seq_length, n_experts
            predicted_expert_mask = predicted_expert_mask.view(-1, predicted_expert_mask.size(-1)) #n_layers*batch_size*seq_length, n_experts
            

            #compute accuracy between target mask and predicted mask
            layer_accuracy =torch.sum(target_expert_mask*predicted_expert_mask, dim = 1)/torch.sum(target_expert_mask, dim = 1)
            mean_accuracy = torch.mean(layer_accuracy)
            total_acc.append(mean_accuracy)
            accelerator.log({"final_test/accuracy": mean_accuracy.item()})
            pbar_test.update(1)
        mean_accuracy = torch.stack(total_acc).mean()
        accelerator.log({"final_test/mean_accuracy": mean_accuracy.item()})

    accelerator.end_training()

def run_test_batches(BERTpredictor, neobert_model, test_dataloader, accelerator, cfg, max_batches=None):
    BERTpredictor.eval()
    with torch.no_grad():
        pbar_test = tqdm(
            test_dataloader,
            desc="Eval",
            unit="batch",
            disable=(cfg.trainer.disable_tqdm or not accelerator.is_main_process),
        )
        stored_batch = {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
        }
        total_acc = []
        for i, batch in enumerate(test_dataloader):
            if max_batches is not None and i >= max_batches:
                break

            if batch["input_ids"].shape[0] != cfg.dataloader.train.batch_size:
                batch, stored_batch = to_target_batch_size(batch, stored_batch, cfg.dataloader.train.batch_size)

            if batch["input_ids"].shape[0] < cfg.dataloader.train.batch_size:
                stored_batch = batch
                continue

            #target
            model_output = neobert_model(batch["input_ids"], batch.get("attention_mask", None), output_expert_usage_loss=False, output_router_logits=True)
            target_gate_logits = model_output['router_logits'] 
            target_expert_mask = get_expert_mask(target_gate_logits, cfg)

            #prediction
            predicted_routed_logits = BERTpredictor(batch["input_ids"])# should we  add the attention mask and make it a normal one not additive?
            predicted_expert_mask = get_expert_mask(predicted_routed_logits, cfg)

            #ignore padded tokens
            pad_mask = batch.get("attention_mask", None)
            if pad_mask is not None:
                pad_mask = pad_mask.view(-1, 1)
                pad_mask = (pad_mask != float("-inf")).squeeze(-1)
                predicted_expert_mask = predicted_expert_mask[:,pad_mask,:]
                target_expert_mask = target_expert_mask[:,pad_mask,:]

            #reshape
            target_expert_mask = target_expert_mask.view(-1, target_expert_mask.size(-1))
            predicted_expert_mask = predicted_expert_mask.view(-1, predicted_expert_mask.size(-1))

            #compute accuracy between target mask and predicted mask
            layer_accuracy = torch.sum(target_expert_mask * predicted_expert_mask, dim=1) / torch.sum(target_expert_mask, dim=1)
            mean_accuracy = torch.mean(layer_accuracy)
            total_acc.append(mean_accuracy)
            #accelerator.log({"test/accuracy": mean_accuracy.item()})
            pbar_test.update(1)
        if total_acc:
            mean_accuracy = torch.stack(total_acc).mean()
            accelerator.log({"test/mean_accuracy": mean_accuracy.item()})
        pbar_test.close()




