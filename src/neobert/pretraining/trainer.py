import os
import shutil
import re
from tqdm import tqdm

from omegaconf import OmegaConf, DictConfig

import datetime

# PyTorch
import torch
#from torch.nn import CrossEntropyLoss

# Hugging Face
from datasets import load_from_disk
from transformers import BatchEncoding
from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from accelerate.utils import DistributedDataParallelKwargs

# Deepspeed
#from deepspeed.utils import safe_get_full_fp32_param

# Our metric object and model
from .metrics import Metrics
from ..model import NeoBERTLMHead, NeoBERTConfig
from ..tokenizer import get_tokenizer
from ..optimizer import get_optimizer
from ..scheduler import get_scheduler
from ..dataloader import get_dataloader
from ..datasetCRAMMING import get_datasetCRAMMING, get_tokenizerCRAMMING
from ..dataset import get_dataset
from ..dataloaderCRAMMING import get_dataloaderCRAMMING

#loss functions
from .losses import mop_loss_fn, hetero_moe_loss_fn, homo_moe_loss_fn, mop_loss_fn_alt
from .analysis import get_normalised_expert_usage_cost_per_sequence,get_entropy,get_mse_per_sequence

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

def trainer(cfg: DictConfig):
    # Get the last checkpoint id
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = os.path.join(cfg.trainer.dir,cfg.model.type +"_"+time_str)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    model_checkpoint_dir = os.path.join(model_dir, "model_checkpoints")
    if cfg.trainer.save_model:
        os.makedirs(model_checkpoint_dir, exist_ok=True)

    iteration = 0
    if cfg.trainer.resume and os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        # This regular expression was taken from accelerator.load_state()
        folders = os.listdir(checkpoint_dir)
        iteration = max(int(re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)[0]) for folder in folders) + 1

    # Accelerator object
    project_config = ProjectConfiguration(
        model_dir,
        automatic_checkpoint_naming=True,
        total_limit=cfg.trainer.accelerate.max_ckpt,
        iteration=iteration,
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,  # enable manual control of the scheduler
        mixed_precision=cfg.trainer.mixed_precision,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        log_with="wandb",
        project_config=project_config,
        kwargs_handlers=[kwargs],
    )

    #wandb_name = f"alpha:{cfg.model.loss.cost_based_loss_alpha}_n_samples:{cfg.dataset.train.num_samples}"

    # Initialise the wandb run and pass wandb parameters
    os.makedirs(cfg.wandb.dir, exist_ok=True)
    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        init_kwargs={
            "wandb": {
                "name": cfg.wandb.name,
                "entity": cfg.wandb.entity,
                "config": OmegaConf.to_container(cfg) | {"distributed_type": accelerator.distributed_type},
                "tags": cfg.wandb.tags,
                "dir": cfg.wandb.dir,
                "mode": cfg.wandb.mode,
                "resume": cfg.wandb.resume,
            }
        },
    )

    # Set the seed
    set_seed(cfg.seed)

    # Enable TF32 on matmul and on cuDNN
    torch.backends.cuda.matmul.allow_tf32 = cfg.trainer.tf32
    torch.backends.cudnn.allow_tf32 = cfg.trainer.tf32

    # Local and global counters
    metrics = Metrics()
    accelerator.register_for_checkpointing(metrics)

    # Get the dtype for the pad_mask
    dtype_pad_mask = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype_pad_mask = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype_pad_mask = torch.bfloat16

    if cfg.dataset.name == "crammingpile": #careful to have correct configs in pretraining.yaml cfg such that cfg.dataloader is CRAMMINGdataloader,  cfg.tokenizer is CRAMMINGtokenizer cfg.datacollator is CRAMMINGmlm_15
        # Tokenizer
        tokenizer = get_tokenizerCRAMMING(cfg.tokenizer.tokenizer_parent_dir) 

        #Dataset
        train_dataset = get_datasetCRAMMING( 
            **cfg.dataset.train)

        #Dataloader
        dataloader_config_args = dict(**cfg.dataloader.train, **cfg.datacollator)
        dataloader_config_args["shuffle"] = not cfg.dataset.train.streaming
        train_dataloader = get_dataloaderCRAMMING(train_dataset, tokenizer, **dataloader_config_args)
    
 
        #train_dataloader = get_dataloaderCRAMMING(train_dataset, tokenizer, **cfg.dataloader.train, **cfg.datacollator)


    else:

        # Tokenizer
        tokenizer = get_tokenizer(**cfg.tokenizer) 

        #Dataset
        train_dataset = get_dataset(cfg, 
            **cfg.dataset.train)
        
        # Dataloader
        train_dataloader = get_dataloader(train_dataset, tokenizer, dtype=dtype_pad_mask, **cfg.dataloader.train, **cfg.datacollator)

        # # Tokenizer
        # tokenizer = get_tokenizer(**cfg.tokenizer)

        # # Dataset
        # train_dataset = load_from_disk(cfg.dataset.path_to_disk)

        # # Dataloader
        # train_dataloader = get_dataloader(train_dataset, tokenizer, dtype=dtype_pad_mask, **cfg.dataloader.train, **cfg.datacollator)

    # Model
    
    model = NeoBERTLMHead(NeoBERTConfig(**cfg.model, **cfg.tokenizer, pad_token_id=tokenizer.pad_token_id))
    accelerator.log({"model_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})

    # Optimizer and Scheduler
    optimizer = get_optimizer(model, accelerator.distributed_type, name=cfg.optimizer.name, **cfg.optimizer.hparams)
    scheduler = get_scheduler(optimizer=optimizer, lr=cfg.optimizer.hparams.lr, **cfg.scheduler)

    # Prepare with accelerate
    train_dataloader, model, optimizer, scheduler = accelerator.prepare(
        train_dataloader,
        model,
        optimizer,
        scheduler,
    )

    #compile after preparation with accelerate
    model = torch.compile(model)

    # Resume from the latest checkpoint
    skipped_train_dataloader = None
    if cfg.trainer.resume and os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        accelerator.load_state()
        train_dataloader.set_epoch(metrics["train/epochs"])
        skipped_train_dataloader = accelerator.skip_first_batches(train_dataloader, metrics["train/batches"] % len(train_dataloader))

    # Progress bar
    pbar = tqdm(
        desc="Train",
        unit="step",
        initial=metrics["train/steps"],
        total=cfg.trainer.max_steps,
        disable=(cfg.trainer.disable_tqdm or not accelerator.is_main_process),
    )

    #create directory to store model
    if cfg.trainer.save_model:
        if accelerator.distributed_type is DistributedType.DEEPSPEED:
            print("this part of the code needs to be updated")
            model.save_checkpoint(model_dir, tag=metrics["train/steps"])
        else:
            os.makedirs(model_dir, exist_ok=True)
            OmegaConf.save(cfg, os.path.join(model_dir, "config.yaml"))

    # Add buffer for moving variance
    moving_mean_buffer = []

    # Buffers for correlation analysis
    mse_loss_buffer = []
    expert_usage_buffer = []
    buffer_size = 100

    while cfg.trainer.max_steps > metrics["train/steps"]:
        # Use skipped_train_dataloader the first epoch after resuming
        dataloader = train_dataloader if skipped_train_dataloader is None else skipped_train_dataloader

        stored_batch = {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
        }
        i = 0
        for batch in dataloader:
            # Update number of batches
            
            metrics["train/batches"] += 1
            i += 1

            # Pack or truncate the batch to target batch size (batch size might be variable due to sequence packing).
            if batch["input_ids"].shape[0] != cfg.dataloader.train.batch_size:
                batch, stored_batch = to_target_batch_size(batch, stored_batch, cfg.dataloader.train.batch_size)

            # If it is still smaller, stored batches were not enough and we skip to the next iteration to fill the batch
            if batch["input_ids"].shape[0] < cfg.dataloader.train.batch_size:
                stored_batch = batch
                continue

            # Under the no_sync context manager, PyTorch will skip synchronizing the gradients when .backward() is
            # called, and the first call to .backward() outside this context manager will trigger the synchronization.
            # Accumulating manually gives more flexibility and is compatible with TPUs.
            if metrics["train/batches"] % cfg.trainer.gradient_accumulation_steps != 0:
                with accelerator.no_sync(model):
                    # Forward pass

                
                    
                    if cfg.model.type == "homo_moe":
                        model_output = model(batch["input_ids"], batch.get("attention_mask", None), output_expert_usage_loss=False, output_router_logits=True)
                        train_loss, mlm_loss, load_balancing_loss = homo_moe_loss_fn(model_output['logits'], model_output['router_logits'], cfg, batch)
                    if cfg.model.type == "hetero_moe":
                        model_output = model(batch["input_ids"], batch.get("attention_mask", None), output_expert_usage_loss=False, output_router_logits=True)
                        train_loss, penalty_loss, entropy_loss, mean_num_activated_experts = hetero_moe_loss_fn(model_output['logits'], model_output['router_logits'], cfg, batch)
                    if cfg.model.type == "mop":
                        num_masked_tokens = cfg.dataloader.train.batch_size * cfg.tokenizer.max_length * metrics["train/steps"]*cfg.datacollator.mlm_probability
                        model_output = model(batch["input_ids"], batch.get("attention_mask", None), output_expert_usage_loss=True, output_router_logits=False)
                        train_loss,mlm_loss,expert_loss, cost_based_loss_alpha = mop_loss_fn(model_output['logits'], model_output['expert_usage_loss'], cfg, batch, num_masked_tokens)
                        #train_loss,mlm_loss,expert_loss,cost_based_loss_alpha = mop_loss_fn_alt(model_output['logits'], model_output['router_logits'], cfg, batch, num_masked_tokens) 


                    logits = model_output['logits']
                    # Compute gradient
                    # if metrics["train/steps"] % 2== 0:
                    #     accelerator.backward(mlm_loss)
                    # else:
                    #     accelerator.backward(load_balancing_loss)

                    accelerator.backward(train_loss)
                    #print(cost_based_loss_alpha)


                    # Log metrics
                    metrics["train/local_samples"] += batch["input_ids"].shape[0]
                    if "attention_mask" in batch.keys():
                        metrics["train/local_tokens"] += (batch["attention_mask"] == 0).sum().item()
                    else:
                        metrics["train/local_tokens"] += batch["input_ids"].shape[1]
                    metrics["train/local_num_pred"] += (batch["labels"] != -100).sum().item()
                    metrics["train/local_sum_loss"] += train_loss.item() * (batch["labels"] != -100).sum().item()
                    metrics["train/local_sum_mlm_loss"] += mlm_loss.item() * (batch["labels"] != -100).sum().item()
                    metrics["train/local_num_correct"] += (logits.argmax(dim=-1) == batch["labels"]).sum().item()
                    if cfg.model.type == "mop":
                        metrics["train/local_sum_expert_loss"] += expert_loss.item() * (batch["labels"] != -100).sum().item()
                        metrics["train/cost_based_loss_alpha"] = cost_based_loss_alpha.item()
                        print("hello:", cost_based_loss_alpha.item())
                        
                    if cfg.model.type == "hetero_moe":
                        metrics["train/local_sum_penalty_loss"] += penalty_loss.item() * (batch["labels"] != -100).sum().item()
                        metrics["train/local_sum_entropy_loss"] += entropy_loss.item() * (batch["labels"] != -100).sum().item()
                    if cfg.model.type =="homo_moe":
                        metrics["train/local_sum_load_balancing_loss"] += load_balancing_loss.item() * (batch["labels"] != -100).sum().item()

            else:
                # Forward pass
                if cfg.model.type == "homo_moe":
                    model_output = model(batch["input_ids"], batch.get("attention_mask", None), output_expert_usage_loss=False, output_router_logits=True)
                    train_loss, mlm_loss, load_balancing_loss = homo_moe_loss_fn(model_output['logits'], model_output['router_logits'], cfg, batch)
                if cfg.model.type == "hetero_moe":
                    model_output = model(batch["input_ids"], batch.get("attention_mask", None), output_expert_usage_loss=False, output_router_logits=True)
                    train_loss, mlm_loss,penalty_loss, entropy_loss, mean_num_activated_experts = hetero_moe_loss_fn(model_output['logits'], model_output['router_logits'], cfg, batch)
                if cfg.model.type == "mop":
                    num_masked_tokens = cfg.dataloader.train.batch_size * cfg.tokenizer.max_length * metrics["train/steps"]*cfg.datacollator.mlm_probability
                    model_output = model(batch["input_ids"], batch.get("attention_mask", None), output_expert_usage_loss=True, output_router_logits=True)
                    train_loss,mlm_loss,expert_loss,cost_based_loss_alpha = mop_loss_fn(model_output['logits'], model_output['expert_usage_loss'], cfg, batch, num_masked_tokens) 
                    
                    #ANALYSING CORRELATION BETWEEN EXPERT USAGE AND LOSS AND MSE LOSS PER SEQUENCE

                    #compute variance of expert loss between sequences within a batch and  across batches

                    normalised_expert_usage_cost_per_seq = get_normalised_expert_usage_cost_per_sequence(model_output['router_logits'], batch.get("attention_mask", None), cfg)
                    var_across_sequences = torch.var(normalised_expert_usage_cost_per_seq)
                    metrics["train/var_across_sequences"] = var_across_sequences.item()
                    mean_normalised_expert_usage_cost_per_batch = normalised_expert_usage_cost_per_seq.mean()

                    # Update moving buffer and compute moving variance
                    moving_mean_buffer.append(mean_normalised_expert_usage_cost_per_batch.item())
                    if len(moving_mean_buffer) > 10:
                        moving_mean_buffer.pop(0)
                    if len(moving_mean_buffer) > 1:
                        moving_var = torch.tensor(moving_mean_buffer).var(unbiased=False).item()
                    else:
                        moving_var = 0.0
                    metrics["train/moving_var_mean_normalised_expert_usage_cost_per_batch"] = moving_var

                    #compute total entropy
                    entropy = get_entropy(model_output['router_logits'], cfg, batch.get("attention_mask", None))
                    metrics["train/entropy"] = entropy.item()

                    #per say correlation computation
                    # Extract per-sequence metrics
                    
                    normalised_expert_usage_cost_per_seq = get_normalised_expert_usage_cost_per_sequence(model_output['router_logits'], batch.get("attention_mask", None), cfg)
                    mse_loss_per_seq = get_mse_per_sequence(model_output['logits'], cfg,batch)

                    # Accumulate in buffers
                    expert_usage_buffer.extend(normalised_expert_usage_cost_per_seq.detach().cpu().tolist())
                    mse_loss_buffer.extend(mse_loss_per_seq.detach().cpu().tolist())

                    # When buffer is full, compute correlation and log, then reset
                    if len(expert_usage_buffer) >= buffer_size and len(mse_loss_buffer) >= buffer_size:
                        # Truncate to buffer_size in case of overflow
                        expert_usage_arr = torch.tensor(expert_usage_buffer[:buffer_size])
                        mse_loss_arr = torch.tensor(mse_loss_buffer[:buffer_size])
                        # Compute Pearson correlation
                        if expert_usage_arr.std() > 0 and mse_loss_arr.std() > 0:
                            correlation = torch.corrcoef(torch.stack([expert_usage_arr, mse_loss_arr]))[0, 1].item()
                        else:
                            correlation = 0.0
                        metrics["train/expert_usage_mse_corr"] = correlation
                        # Log correlation
                        accelerator.log({"train/expert_usage_mse_corr": correlation})
                        # Reset buffers
                        expert_usage_buffer = []
                        mse_loss_buffer = []
                    



                logits = model_output['logits']

                
                # if metrics["train/steps"] % 2== 0:
                #         accelerator.backward(mlm_loss)
                # else:
                #     accelerator.backward(expert_loss)
                # Compute gradient and apply clipping                    
                accelerator.backward(train_loss)
                # accelerator.backward(load_balancing_loss)

                if cfg.trainer.gradient_clipping is not None and cfg.trainer.gradient_clipping > 0:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clipping)


                # #DEBUG: Check for "infinite" or extremely large gradients
                # large_grad_threshold = 1e20  # You can adjust this threshold if needed
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         if torch.isinf(param.grad).any():
                #             print(f"[GRAD INF] Layer '{name}' has inf gradients")
                #         elif (param.grad.abs() >= large_grad_threshold).any():
                #             max_val = param.grad.abs().max().item()
                #             print(f"[GRAD HUGE] Layer '{name}' has gradient magnitude up to {max_val:.3e}")

                # Log metrics
                pbar.update(1)
                metrics["train/steps"] += 1
                metrics["train/local_samples"] += batch["input_ids"].shape[0]
                if "attention_mask" in batch.keys():
                    metrics["train/local_tokens"] += (batch["attention_mask"] == 0).sum().item()
                else:
                    metrics["train/local_tokens"] += batch["input_ids"].shape[1]
                metrics["train/local_num_pred"] += (batch["labels"] != -100).sum().item()
                metrics["train/local_sum_loss"] += train_loss.item() * (batch["labels"] != -100).sum().item()
                metrics["train/local_sum_mlm_loss"] += mlm_loss.item() * (batch["labels"] != -100).sum().item()
                metrics["train/local_num_correct"] += (logits.argmax(dim=-1) == batch["labels"]).sum().item()

                if cfg.model.type == "mop":
                    metrics["train/local_sum_expert_loss"] += expert_loss.item() * (batch["labels"] != -100).sum().item()
                    metrics["train/cost_based_loss_alpha"] = cost_based_loss_alpha.item()
                if cfg.model.type == "hetero_moe":
                    metrics["train/local_sum_penalty_loss"] += penalty_loss.item() * (batch["labels"] != -100).sum().item()
                    metrics["train/local_sum_entropy_loss"] += entropy_loss.item() * (batch["labels"] != -100).sum().item()
                    metrics["train/mean_num_activated_experts"] = mean_num_activated_experts.item()
                if cfg.model.type =="homo_moe":
                    metrics["train/local_sum_load_balancing_loss"] += load_balancing_loss.item() * (batch["labels"] != -100).sum().item()

                # Update the parameters and the scheduler
                optimizer.step()
                scheduler.step()

                if metrics["train/steps"] % cfg.wandb.log_interval == 0:


                    # https://deepspeed.readthedocs.io/en/latest/zero3.html#deepspeed.utils.safe_get_full_grad
                    if accelerator.distributed_type is DistributedType.DEEPSPEED:
                        metrics["train/grad_norm"] = model.get_global_grad_norm()
                        metrics["train/weight_norm"] = (
                            sum([safe_get_full_fp32_param(p).norm(2) ** 2 for p in model.parameters()if p.grad is not None]) ** 0.5
                        ).item()
                    # DDP
                    else:
                        metrics["train/grad_norm"] = (sum([p.grad.norm(2) ** 2 for p in model.parameters() if p.grad is not None]) ** 0.5).item()
                        metrics["train/weight_norm"] = (sum([p.norm(2) ** 2 for p in model.parameters() if p.grad is not None]) ** 0.5).item()

                        # plot gradients of the gates and of the experts
                        gate_grad_norm = 0.0
                        expert_grad_norm = 0.0
                        embedding_grad_norm = 0.0
                        attention_grad_norm = 0.0

                        num_gate_params = 0.0
                        num_expert_params = 0.0
                        num_embedding_params = 0.0
                        num_attention_params = 0.0



                        for name, param in model.named_parameters():
                            # print(name)
                            if param.grad is not None:
                                if "gate" in name:
                                    gate_grad_norm += param.grad.norm(2).item() ** 2
                                    num_gate_params += param.numel()

                                if "experts" in name:
                                    expert_grad_norm += param.grad.norm(2).item() ** 2
                                    num_expert_params += param.numel()
                                    # print(num_expert_params)
                                if "model.encoder" in name or "decoder" in name or "model.positional_embedding" in name:
                                    embedding_grad_norm += param.grad.norm(2).item() ** 2
                                    num_embedding_params += param.numel()
                                if "qkv" in name or "wo" in name:
                                    attention_grad_norm  += param.grad.norm(2).item() ** 2
                                    num_attention_params += param.numel()

                        

                        total_num_params = num_gate_params + num_expert_params + num_embedding_params + num_attention_params
                        total_grads = (gate_grad_norm + expert_grad_norm + embedding_grad_norm + attention_grad_norm) ** 0.5
                        metrics["train/rel_prop_gate_grad_norm"] = (gate_grad_norm ** 0.5)/total_grads * (total_num_params / num_gate_params)
                        metrics["train/rel_prop_expert_grad_norm"] = (expert_grad_norm ** 0.5)/total_grads * (total_num_params / num_expert_params)
                        metrics["train/rel_prop_embedding_grad_norm"] = (embedding_grad_norm ** 0.5)/total_grads * (total_num_params / num_embedding_params)
                        metrics["train/rel_prop_attention_grad_norm"] = (attention_grad_norm ** 0.5)/total_grads * (total_num_params / num_attention_params)

                        metrics["train/prop_gate_grad_norm"] = (gate_grad_norm ** 0.5)/total_grads
                        metrics["train/prop_expert_grad_norm"] = (expert_grad_norm ** 0.5)/total_grads 
                        metrics["train/prop_embedding_grad_norm"] = (embedding_grad_norm ** 0.5)/total_grads 
                        metrics["train/prop_attention_grad_norm"] = (attention_grad_norm ** 0.5)/total_grads 

                        #special metrics when we specifically look at both loss gradients.

                        # if metrics["train/steps"] % 2 == 1:
                        #     metrics["train/gate_grads_norm_mlm"] = gate_grad_norm
                        #     metrics["train/expert_grads_norm_mlm"] = expert_grad_norm
                        #     metrics["train/embedding_grads_norm_mlm"] = embedding_grad_norm
                        #     metrics["train/attention_grads_norm_mlm"] = attention_grad_norm
                        # else:
                        #     metrics["train/gate_grads_norm_other"] += gate_grad_norm
                        #     metrics["train/expert_grads_norm_other"] += expert_grad_norm
                        #     metrics["train/embedding_grads_norm_other"] += embedding_grad_norm
                        #     metrics["train/attention_grads_norm_other"] += attention_grad_norm

                    metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]
                    metrics.log(accelerator,cfg.model.type)

                # Save the accelerator state from the main process
                if metrics["train/steps"] % cfg.trainer.accelerate.save_steps == 0:
                    accelerator.save_state()

                # Save the pytorch model
                if metrics["train/steps"] % cfg.trainer.model.save_steps == 0:
                    if cfg.trainer.model.max_ckpt is not None:
                        # Delete checkpoints if there are too many
                        files = os.listdir(model_checkpoint_dir)
                        iterations = [int(f) for f in files if f.isdigit()]
                        iterations.sort()

                        # Remove files with the smallest iterations until the limit is met
                        while iterations is not None and len(iterations) >= cfg.trainer.model.max_ckpt:
                            file_to_remove = iterations.pop(0)
                            shutil.rmtree(os.path.join(model_checkpoint_dir, str(file_to_remove)))
                            print(
                                f"Deleted old model checkpoint {file_to_remove} due to limit " f"(max_ckpt = {cfg.trainer.model.max_ckpt})"
                            )
                    # Save the checkpoint
                    if cfg.trainer.save_model:
                        # Get current time
                        if accelerator.distributed_type is DistributedType.DEEPSPEED:
                            model.save_checkpoint(model_checkpoint_dir, tag=metrics["train/steps"])
                        else:
                            path = os.path.join(model_checkpoint_dir,str(metrics["train/steps"]))
                            os.makedirs(path, exist_ok=True)
                            torch.save(
                                model.state_dict(),
                                os.path.join(path, "state_dict.pt"),
                            )

                if metrics["train/steps"] >= cfg.trainer.max_steps:
                    break

                # Reset the gradient
                optimizer.zero_grad()

        # Log metrics
        metrics["train/epochs"] += 1

        # "Remove" the skipped dataloader once exhausted
        skipped_train_dataloader = None

    # Make sure that the wandb tracker finishes correctly and close the progress bar
    pbar.close()
    accelerator.end_training()
