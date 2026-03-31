# This script visualises the routing paths of tokens through the experts in a MoP model.
# It loads a pretrained NeoBERT model, runs it on a test dataset, and visualises 
# which experts are used for each token in the input. 
# The visualisation is logged to Weights & Biases (wandb) for easy inspection.


from datetime import datetime
from ..tokenizer import get_tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..dataloader import get_dataloader

from .analysis import AnalysisLogger

from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from accelerate.utils import DistributedDataParallelKwargs

from neobert.utils import to_target_batch_size, _resolve_mixed_precision, _get_pretrained_neobert_model
from .analysis_utils import get_expert_mask
from ..dataset import get_dataset
from omegaconf import OmegaConf
import os
import datetime

import wandb

def visualise_pathways(cfg_visualise_pathways):

    # load  pretrained neobert config
    cfg_path = os.path.join(cfg_visualise_pathways.saved_model.base_path, "config.yaml")
    cfg_neobert = OmegaConf.load(cfg_path)

    # Resolve mixed precision based on hardware availability
    resolved_mixed_precision = _resolve_mixed_precision(cfg_neobert.trainer.mixed_precision)
    
    # SET UP ACCELERATOR---------------------------
    
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,  # enable manual control of the scheduler
        mixed_precision=resolved_mixed_precision,
        gradient_accumulation_steps=cfg_neobert.trainer.gradient_accumulation_steps,
        log_with="wandb",
        # project_config=project_config,
        kwargs_handlers=[kwargs],
    )

    # Initialise the wandb run and pass wandb parameters
    os.makedirs(cfg_visualise_pathways.wandb.dir, exist_ok=True)
    accelerator.init_trackers(
        project_name=cfg_visualise_pathways.wandb.project,
        init_kwargs={
            "wandb": {
                "name": cfg_visualise_pathways.wandb.name,
                "entity": cfg_visualise_pathways.wandb.entity,
                "config": OmegaConf.to_container(cfg_visualise_pathways)
                | {"distributed_type": accelerator.distributed_type},
                "tags": cfg_visualise_pathways.wandb.tags,
                "dir": cfg_visualise_pathways.wandb.dir,
                "mode": cfg_visualise_pathways.wandb.mode,
                "resume": cfg_visualise_pathways.wandb.resume,
            }
        },
    )

    # Set seed for reproducibility
    set_seed(25)
    g = torch.Generator()
    g.manual_seed(25)

    # Enable TF32 on matmul and on cuDNN
    torch.backends.cuda.matmul.allow_tf32 = cfg_neobert.trainer.tf32
    torch.backends.cudnn.allow_tf32 = cfg_neobert.trainer.tf32

    # Get the dtype for the pad_mask
    dtype_pad_mask = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype_pad_mask = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype_pad_mask = torch.bfloat16

    # END SET UP ACCELERATOR---------------------------

    # device=torch.device("cuda" if torch.cuda.is_available() else "cpu") # en ai-je encore besoin?

    # wandb.init(name=cfg_visualise_pathways.wandb.name,
    #             project=cfg_visualise_pathways.wandb.project,
    #               entity = cfg_visualise_pathways.wandb.entity,
    #               config=OmegaConf.to_container(cfg_visualise_pathways, resolve=True))


    # set wandb run name to include model type and timestamp
    base_name = cfg_neobert.model.type + "_visualiser"
    if cfg_neobert.model.type == "mop":
        base_name = base_name + "_" + str(cfg_neobert.model.loss.cost_based_loss_alpha_end)
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.run.name = base_name + "_" + time_str

    # get pretrained neobert model
    tokenizer = get_tokenizer(**cfg_neobert.tokenizer)
    neobert_model = _get_pretrained_neobert_model(cfg_neobert, cfg_visualise_pathways, pad_token_id=tokenizer.pad_token_id)

    # set to eval mode
    neobert_model.eval()
    
    # get dataset
    test_dataset = get_dataset(cfg_neobert, **cfg_visualise_pathways.dataset.test)

    # build dataloader
    test_dataloader = get_dataloader(
        test_dataset,
        tokenizer,
        dtype=dtype_pad_mask,
        **cfg_visualise_pathways.dataloader.test,
        **cfg_visualise_pathways.datacollator,
    )

    # Prepare with accelerate
    (
        test_dataloader,
        neobert_model,
    ) = accelerator.prepare(
        test_dataloader,
        neobert_model,
    )


    with torch.no_grad():

        stored_batch = {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
        }
        max_batches=cfg_visualise_pathways.num_batches_to_visualise

        printed_tokens = 0  # Track how many tokens we've printed
        max_print_tokens = 1
        for i, batch in enumerate(test_dataloader):
            #we go through a dataset instead of generating random tokens from thin air  
            # because it would take more effort to create a well formated batch from this
            if max_batches is not None and i >= max_batches:
                break

            if batch["input_ids"].shape[0] != cfg_visualise_pathways.dataloader.test.batch_size:
                batch, stored_batch = to_target_batch_size(
                    batch, stored_batch, cfg_visualise_pathways.dataloader.test.batch_size
                )
            if batch["input_ids"].shape[0] < cfg_visualise_pathways.dataloader.test.batch_size:
                stored_batch = batch
                continue

            # target
            model_output = neobert_model(
                batch["input_ids"],
                batch.get("attention_mask", None),
                output_expert_usage_loss=False,
                output_router_logits=True,
            )
            target_gate_logits = model_output["router_logits"]
            target_expert_mask = get_expert_mask(
                target_gate_logits, routing_strategy="top_k"
            )

            # ignore padded tokens
            pad_mask = batch.get("attention_mask", None)
            if pad_mask is not None:
                pad_mask = pad_mask.view(-1, 1)
                pad_mask = (pad_mask != float("-inf")).squeeze(-1)
                target_expert_mask = target_expert_mask[:, pad_mask, :]

            # visualisation:

            analyser = AnalysisLogger(cfg_neobert,accelerator)
            metrics_dict = {}
            analyser.visualise_token_routing_paths(
                batch,target_gate_logits,metrics_dict, cfg_visualise_pathways.max_print_tokens_per_batch)
            
            accelerator.log(metrics_dict)
            




           
           
