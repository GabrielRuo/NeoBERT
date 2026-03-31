import os
import random

import numpy as np
import wandb
from collections import defaultdict
from transformers import BertModel, BatchEncoding


import datetime
from ..tokenizer import get_tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..dataloader import get_dataloader
from ..model import NeoBERTLMHead, NeoBERTConfig

# from ..pretraining import mop_loss_fn_balanced
from .analysis import AnalysisTestTrainedModel
from ..optimizer import get_optimizer
from ..scheduler import get_scheduler

from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from accelerate.utils import DistributedDataParallelKwargs

from ..dataset import get_dataset
from omegaconf import OmegaConf, DictConfig


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
            tmp[key] = torch.split(
                batch[key], [target_size, batch_size - target_size], dim=0
            )
            batch[key] = tmp[key][0]
            stored_batch[key] = (
                tmp[key][1]
                if stored_batch[key] is None
                else torch.cat([tmp[key][1], stored_batch[key]], dim=0)
            )

    # If the batch is to small, we fetch stored samples
    elif batch_size < target_size and stored_batch["input_ids"] is not None:
        stored_batch_size = stored_batch["input_ids"].shape[0]
        missing = target_size - batch_size

        # Fetch only necessary samples if storage is larger than required
        if missing < stored_batch_size:
            for key in batch.keys():
                stored_batch[key].to(batch[key].device)
                tmp[key] = torch.split(
                    stored_batch[key], [missing, stored_batch_size - missing], dim=0
                )
                batch[key] = torch.cat([batch[key], tmp[key][0]], dim=0)
                stored_batch[key] = tmp[key][1]
                stored_batch[key].to("cpu", non_blocking=True)

        # Concatenate otherwise
        else:
            for key in batch.keys():
                batch[key] = torch.cat([batch[key], stored_batch[key]], dim=0)
                stored_batch[key] = None

    return batch, stored_batch


def pretrained_model_tester(cfg_test):

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,  # enable manual control of the scheduler
        mixed_precision=cfg_test.trainer.mixed_precision,
        gradient_accumulation_steps=cfg_test.trainer.gradient_accumulation_steps,
        log_with="wandb",
        # project_config=project_config,
        kwargs_handlers=[kwargs],
    )

    # Initialise the wandb run and pass wandb parameters
    os.makedirs(cfg_test.wandb.dir, exist_ok=True)
    accelerator.init_trackers(
        project_name=cfg_test.wandb.project,
        init_kwargs={
            "wandb": {
                "name": cfg_test.wandb.name,
                "entity": cfg_test.wandb.entity,
                "config": OmegaConf.to_container(cfg_test)
                | {"distributed_type": accelerator.distributed_type},
                "tags": cfg_test.wandb.tags,
                "dir": cfg_test.wandb.dir,
                "mode": cfg_test.wandb.mode,
                "resume": cfg_test.wandb.resume,
            }
        },
    )
    set_seed(25)

    # Enable TF32 on matmul and on cuDNN
    torch.backends.cuda.matmul.allow_tf32 = cfg_test.trainer.tf32
    torch.backends.cudnn.allow_tf32 = cfg_test.trainer.tf32

    # Get the dtype for the pad_mask
    dtype_pad_mask = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype_pad_mask = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype_pad_mask = torch.bfloat16

    print("dtype for pad_mask:", dtype_pad_mask)

    # END SET UP ACCELERATOR---------------------------

    # load  pretrained config
    cfg_path = os.path.join(cfg_test.saved_model.base_path, "config.yaml")
    cfg = OmegaConf.load(cfg_path)

    base_name = cfg.model.type + "_tester"
    if cfg.model.type == "mop":
        base_name = base_name + "_" + str(cfg.model.loss.cost_based_loss_alpha_end)
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.run.name = base_name + "_" + time_str

    # get pretrained neobert model
    tokenizer = get_tokenizer(**cfg.tokenizer)
    neobert_model = NeoBERTLMHead(
        NeoBERTConfig(**cfg.model, **cfg.tokenizer, pad_token_id=tokenizer.pad_token_id)
    )
    if "glue" in cfg_test.saved_model.checkpoint:
        # change the way of callin a model if using glue
        state_dict_path = os.path.join(
            cfg_test.saved_model.base_path,
            cfg_test.saved_model.checkpoint,
            "state_dict.pt",
        )
    else:
        state_dict_path = os.path.join(
            cfg_test.saved_model.base_path,
            "model_checkpoints",
            cfg_test.saved_model.checkpoint,
            "state_dict.pt",
        )

    neobert_state_dict = torch.load(state_dict_path, map_location="cpu")

    # Fix keys: strip "_orig_mod." if present
    new_state_dict = {}
    has_pretrained_lm_head = False
    for k, v in neobert_state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod.") :]] = v
        else:
            new_state_dict[k] = v
        if "decoder" in k:
            has_pretrained_lm_head = True

    print("has_pretrained_lm_head:", has_pretrained_lm_head)
    # If the pretrained model does not already have  a pretrained LM head we will keep it randomly initialised. Hence strict = False
    neobert_model.load_state_dict(new_state_dict, strict=False)

    test_dataset = get_dataset(cfg, **cfg_test.dataset.test)

    test_dataloader = get_dataloader(
        test_dataset,
        tokenizer,
        dtype=dtype_pad_mask,
        **cfg_test.dataloader.test,
        **cfg_test.datacollator,
    )
    # Optimizer and Scheduler

    if has_pretrained_lm_head:
        test_dataloader, neobert_model = accelerator.prepare(
            test_dataloader, neobert_model
        )
    if not has_pretrained_lm_head:
        max_steps = 10000
        print(
            f"The pretrained model does not have a pretrained LM head. We will finetune the LM head on the train dataset for  {max_steps} steps."
        )
        train_dataset = get_dataset(cfg, **cfg.dataset.train)

        train_dataloader = get_dataloader(
            train_dataset,
            tokenizer,
            dtype=dtype_pad_mask,
            **cfg.dataloader.train,
            **cfg.datacollator,
        )
        optimizer = get_optimizer(
            neobert_model,
            accelerator.distributed_type,
            name=cfg.optimizer.name,
            **cfg.optimizer.hparams,
        )
        scheduler = get_scheduler(
            optimizer=optimizer, lr=cfg.optimizer.hparams.lr, **cfg.scheduler
        )

        print("Train dataset size:", len(train_dataloader))
        print("Test dataset size:", len(test_dataloader))

        test_dataloader, train_dataloader, neobert_model, optimizer, scheduler = (
            accelerator.prepare(
                test_dataloader, train_dataloader, neobert_model, optimizer, scheduler
            )
        )
        # need to do some finetuning of the head
        print(len(train_dataloader))
        pbar_train = tqdm(
            train_dataloader,
            desc="Finetune LM Head",
            initial=0,
            unit="batch",
            disable=(cfg_test.trainer.disable_tqdm or not accelerator.is_main_process),
        )
        stored_batch = {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
        }

        # initialize metrics to avoid KeyError when using +=
        metrics = {
            "test/local_num_correct": 0,
            "test/local_num_pred": 0,
            "test/accuracy": 0.0,
        }
        max_batches = (
            None  # keep behavior consistent; change if you want to limit batches
        )

        # iterate over the tqdm-wrapped dataloader so progress/display is consistent
        analysis = AnalysisTestTrainedModel(cfg, accelerator, max_steps)
        for i, batch in enumerate(pbar_train):
            if max_batches is not None and i >= max_batches:
                break

            if batch["input_ids"].shape[0] != cfg.dataloader.train.batch_size:
                batch, stored_batch = to_target_batch_size(
                    batch, stored_batch, cfg.dataloader.train.batch_size
                )

            if batch["input_ids"].shape[0] < cfg.dataloader.train.batch_size:
                stored_batch = batch
                continue

            # target
            model_output = neobert_model(
                batch["input_ids"],
                batch.get("attention_mask", None),
                output_expert_usage_loss=True,
                output_router_logits=True,
            )
            logits = model_output["logits"]

            num_masked_tokens = (
                cfg.dataloader.train.batch_size
                * cfg.tokenizer.max_length
                * i
                * cfg.datacollator.mlm_probability
            )  # does not actually matter since we use alpha_scaling = 0
            # train_loss,mlm_loss,expert_loss, load_balancing_loss, cost_based_loss_alpha = mop_loss_fn_balanced(model_output['logits'], model_output['router_logits'],model_output['expert_usage_loss'], cfg, batch, num_masked_tokens)

            # accelerator.backward(train_loss)
            if (
                cfg.trainer.gradient_clipping is not None
                and cfg.trainer.gradient_clipping > 0
            ):
                accelerator.clip_grad_norm_(
                    neobert_model.parameters(), cfg.trainer.gradient_clipping
                )

            optimizer.step()
            scheduler.step()

            # analysis(batch,model_output,i,metrics,max_steps))
            analysis(batch, model_output, i, metrics)

            metrics["ft/local_num_correct"] += (
                (logits.argmax(dim=-1) == batch["labels"]).sum().item()
            )

            metrics["ft/local_num_pred"] += (batch["labels"] != -100).sum().item()

            if metrics["ft/local_num_pred"] > 0:
                metrics["ft/accuracy"] = (
                    metrics["ft/local_num_correct"] / metrics["ft/local_num_pred"]
                )

            print("accuracy:", metrics["test/accuracy"])
            # print(metrics.keys())

            accelerator.log(metrics)

            optimizer.zero_grad()

    neobert_model.eval()
    # print("is it in training mode?", neobert_model.training)

    # this sets self.training = False for all modules recursively  and hence  switche to top-k routing
    with torch.no_grad():
        pbar_test = tqdm(
            test_dataloader,
            desc="Eval",
            initial=0,
            unit="batch",
            disable=(cfg_test.trainer.disable_tqdm or not accelerator.is_main_process),
        )
        stored_batch = {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
        }

        # initialize metrics to avoid KeyError when using +=
        metrics = {
            "test/local_num_correct": 0,
            "test/local_num_pred": 0,
            "test/accuracy": 0.0,
        }
        max_batches = (
            None  # keep behavior consistent; change if you want to limit batches
        )

        # iterate over the tqdm-wrapped dataloader so progress/display is consistent
        max_steps = len(pbar_test)
        analysis = AnalysisTestTrainedModel(cfg, accelerator, max_steps)
        for i, batch in enumerate(pbar_test):
            if max_batches is not None and i >= max_batches:
                break

            if batch["input_ids"].shape[0] != cfg_test.dataloader.test.batch_size:
                batch, stored_batch = to_target_batch_size(
                    batch, stored_batch, cfg_test.dataloader.test.batch_size
                )

            if batch["input_ids"].shape[0] < cfg_test.dataloader.test.batch_size:
                stored_batch = batch
                continue

            # target
            model_output = neobert_model(
                batch["input_ids"],
                batch.get("attention_mask", None),
                output_expert_usage_loss=True,
                output_router_logits=True,
            )
            logits = model_output["logits"]

            # analysis(batch,model_output,i,metrics,max_steps))
            analysis(batch, model_output, i, metrics)

            metrics["test/local_num_correct"] += (
                (logits.argmax(dim=-1) == batch["labels"]).sum().item()
            )

            metrics["test/local_num_pred"] += (batch["labels"] != -100).sum().item()

            if metrics["test/local_num_pred"] > 0:
                metrics["test/accuracy"] = (
                    metrics["test/local_num_correct"] / metrics["test/local_num_pred"]
                )

            print("accuracy:", metrics["test/accuracy"])
            # print(metrics.keys())

            accelerator.log(metrics)

        # We want:
        # - on the test dataset: to see both our performance and the correlations between expert usage and loss
        # - on the test dataset: to see whether we perform with top-2 routing: do we also see correlation  there? comment je  calcule son expert usage?
