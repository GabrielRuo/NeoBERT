#from pyexpat import model

#from NeoBERT.NeoBERT_dev.src.neobert.pretraining import metrics
from datetime import datetime
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
import datetime
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

# Add missing imports
import random
from sklearn.decomposition import PCA
import matplotlib.lines as mlines
from sklearn.preprocessing import StandardScaler


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

    



def get_expert_mask(gate_logits,routing_strategy,num_experts_per_tok_inference=2,min_expert_cumprob_per_token=0.4):

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        stacked_gate_logits = torch.stack([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)#n_layers, batch_size*seq_length, n_experts
        _,_, num_experts = stacked_gate_logits.shape
    routing_weights = torch.nn.functional.softmax(stacked_gate_logits, dim=-1)


    if routing_strategy == "top_k":
        top_k = num_experts_per_tok_inference #if we wanted  to use top_k for heterogeneous moe
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        target_expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts).sum(2)#n_layers, batch_size*seq_length, n_experts
    
    elif routing_strategy == "top_p":
        top_p = min_expert_cumprob_per_token
        sorted_weights, sorted_indices = torch.sort(routing_weights, dim=-1, descending=False)

        cum_probs = sorted_weights.cumsum(dim=-1)
        mask = cum_probs > 1 - top_p

        unsorted_mask = torch.zeros_like(mask, dtype=torch.bool)
        target_expert_mask = unsorted_mask.scatter(dim=-1, index=sorted_indices, src=mask) #n_layers, batch_size*seq_length, n_experts

    return target_expert_mask # n_layers, batch_size*n_seq, n_experts


def get_routing_weight_tensor(gate_logits):

    """Outputs the routing weights in the form of a matrix n_layers, batch_size*n_seq, n_experts"""
    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        stacked_gate_logits = torch.stack([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)#n_layers, batch_size*seq_length, n_experts
        _,_, num_experts = stacked_gate_logits.shape
    routing_weights = torch.nn.functional.softmax(stacked_gate_logits, dim=-1)
    return routing_weights # n_layers, batch_size*n_seq, n_experts

def get_mean_max_weight_per_layer_vector(gate_logits_tensor):
    """Outputs a vector of size n_layers, batch_size*n_seq corresponding to the mean expert selection weight per token per layer"""
    n_experts = gate_logits_tensor.size(-1)
    expert_indices = torch.arange(n_experts, device=gate_logits_tensor.device).float()  # Shape: (n_experts,)
    mean_expert_index_vector = torch.einsum("ijk,k->ij", gate_logits_tensor, expert_indices) # n_layers, batch_size*n_seq
    max_expert_index_vector = torch.argmax(gate_logits_tensor, dim=-1) # n_layers, batch_size*n_seq
    return mean_expert_index_vector, max_expert_index_vector




# def get_expert_mask(gate_logits,routing_strategy,num_experts_per_tok_inference=2,min_expert_cumprob_per_token):

#     if isinstance(gate_logits, tuple):
#         compute_device = gate_logits[0].device
#         stacked_gate_logits = torch.stack([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)#n_layers, batch_size*seq_length, n_experts
#         _,_, num_experts = stacked_gate_logits.shape
#     routing_weights = torch.nn.functional.softmax(stacked_gate_logits, dim=-1)


#     if cfg.model.routing_strategy == "top_k":
#         top_k = cfg.model.num_experts_per_tok_inference #if we wanted  to use top_k for heterogeneous moe
#         _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
#         target_expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts).sum(2)#n_layers, batch_size*seq_length, n_experts
    
#     elif cfg.model.routing_strategy == "top_p":
#         top_p = cfg.model.min_expert_cumprob_per_token
#         sorted_weights, sorted_indices = torch.sort(routing_weights, dim=-1, descending=False)

#         cum_probs = sorted_weights.cumsum(dim=-1)
#         mask = cum_probs > 1 - top_p

#         unsorted_mask = torch.zeros_like(mask, dtype=torch.bool)
#         target_expert_mask = unsorted_mask.scatter(dim=-1, index=sorted_indices, src=mask) #n_layers, batch_size*seq_length, n_experts

#     return target_expert_mask # n_layers, batch_size*n_seq, n_experts


def pathways_analysis(cfg_predictor):


    #SET UP ACCELERATOR---------------------------  
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

    # set_seed(cfg_predictor.seed)
    # set_seed(5)
    # set_seed(6)
    set_seed(25)
    g = torch.Generator()
    g.manual_seed(25)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

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

    base_name = cfg.model.type + "_predictor"
    if cfg.model.type =="mop":
        base_name = base_name + "_" + str(cfg.model.loss.cost_based_loss_alpha_end)
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.run.name = base_name + "_" + time_str


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

    test_dataset = get_dataset(cfg, 
            **cfg_predictor.dataset.test)

    test_dataloader = get_dataloader(
        test_dataset, tokenizer, dtype=dtype_pad_mask,
        **cfg_predictor.dataloader.test, **cfg_predictor.datacollator
    )

    # Prepare with accelerate
    test_dataloader, neobert_model = accelerator.prepare(
        test_dataloader,
        neobert_model,
    )

    metrics_dict = {}
    run_test_batches(neobert_model, test_dataloader, accelerator, cfg,cfg_predictor, metrics_dict, max_batches=None)

def _visualise_token_path(target_gate_logits, token_index, token_str, visualisations_list):
    # Stack logits if tuple
    if isinstance(target_gate_logits, tuple):
        compute_device = target_gate_logits[0].device
        target_stacked_gate_logits = torch.stack([layer_gate.to(compute_device) for layer_gate in target_gate_logits], dim=0)  # n_layers, batch_size*seq_length, n_experts
    else:
        target_stacked_gate_logits = target_gate_logits
    target_routing_weights = torch.nn.functional.softmax(target_stacked_gate_logits, dim=-1)  # n_layers, batch_size*seq_length, n_experts

    # Select the routing weights for the given token index
    # Shape: (n_layers, n_experts)
    target_weights_token = target_routing_weights[:, token_index, :].detach().cpu().numpy()

    n_layers, n_experts = target_weights_token.shape

    # Compute top-2 experts for each layer for target and prediction
    target_top2 = np.argsort(target_weights_token, axis=1)[:, -2:]  # shape (n_layers, 2)

    # Plot side-by-side heatmaps
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(target_weights_token, aspect='auto', cmap='viridis')
    ax.set_title('Target Routing Weights')
    ax.set_xlabel('Expert')
    ax.set_ylabel('Layer')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add black circles for target top-2 experts
    for layer in range(n_layers):
        # Black circles for target
        for expert in target_top2[layer]:
            ax.scatter(expert, layer, s=80, facecolors='none', edgecolors='black', linewidths=2, marker='o', zorder=3)
        
    fig.suptitle(f"Token index: {token_index} ('{token_str}')")
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])


    black_circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label='Target top-2 expert')

    handles = [black_circle]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize='small', frameon=False, bbox_to_anchor=(0.5, 0.01))

    # Save to buffer and append to visualisations_list
    figure_filename = f"token_path_for:{token_index}.png"
    plt.savefig(figure_filename)
    plt.close(fig)

    visualisations_list.append(wandb.Image(figure_filename))



def run_test_batches(neobert_model, test_dataloader, accelerator, cfg,cfg_predictor, metrics_dict, max_batches=None):
    with torch.no_grad():
        pbar_test = tqdm(
            test_dataloader,
            desc="Eval",
            initial = 0,
            unit="batch",
            disable=(cfg_predictor.trainer.disable_tqdm or not accelerator.is_main_process),
        )
        stored_batch = {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
        }
        total_acc = []
        printed_tokens = 0  # Track how many tokens we've printed
        max_print_tokens = 1
        proportions = []  # Store per-token accuracy proportions

        # Aggregation buffers for analysis across multiple batches
        try:
            n_agg_batches = cfg_predictor.analysis.n_agg_batches
        except Exception:
            n_agg_batches = 10
        mean_accum = []  # list of arrays shape (tokens_in_batch, n_layers)
        max_accum = []   # same shape
        full_accum = []  # list of arrays shape (tokens_in_batch, n_layers * n_experts)

        print("n_agg_batches:", n_agg_batches)

        for i, batch in enumerate(test_dataloader):
            if i > n_agg_batches:
                break

            if batch["input_ids"].shape[0] != cfg_predictor.dataloader.train.batch_size:
                batch, stored_batch = to_target_batch_size(batch, stored_batch, cfg_predictor.dataloader.train.batch_size)

            if batch["input_ids"].shape[0] < cfg_predictor.dataloader.train.batch_size:
                stored_batch = batch
                continue

            #target
            model_output = neobert_model(batch["input_ids"], batch.get("attention_mask", None), output_expert_usage_loss=False, output_router_logits=True)
            target_gate_logits = model_output['router_logits'] 

            routing_weights = get_routing_weight_tensor(target_gate_logits)
            mean_expert_index_vector, max_expert_index_vector = get_mean_max_weight_per_layer_vector(routing_weights)   

            # mean_expert_index_vector : shape (n_layers, tokens)
            # transpose to tokens x n_layers and add to accumulators
            mean_tokens_by_layers = mean_expert_index_vector.T.cpu().numpy()  # shape (tokens, n_layers)
            max_tokens_by_layers = max_expert_index_vector.T.cpu().numpy()    # shape (tokens, n_layers)
            mean_accum.append(mean_tokens_by_layers)
            max_accum.append(max_tokens_by_layers)
            # Build full per-token routing-weight vectors: tokens x (n_layers * n_experts)
            # routing_weights has shape (n_layers, tokens, n_experts)
            try:
                rw = routing_weights.detach().cpu()  # n_layers, tokens, n_experts
                tokens_n = rw.shape[1]
                n_layers = rw.shape[0]
                n_experts = rw.shape[2]
                # permute to tokens, n_layers, n_experts then flatten per token
                full_tokens = rw.permute(1, 0, 2).reshape(tokens_n, n_layers * n_experts).numpy()
                full_accum.append(full_tokens)
            except Exception:
                # If anything goes wrong building the full vector, skip it for this batch
                pass

            # When enough batches collected, aggregate and run PCA once
            if len(mean_accum) >= n_agg_batches:
                mean_agg = np.concatenate(mean_accum, axis=0)  # (n_agg_batches * tokens, n_layers)
                max_agg = np.concatenate(max_accum, axis=0)
                print("max agg:", max_agg.shape)

                # Only run analysis & saving on main process
                if accelerator.is_main_process:
                    print("Running PCA aggregation analysis...")
                    try:
                        # choose safe n_components
                        n_samples_mean, n_features_mean = mean_agg.shape
                        n_components_mean = min(2, n_features_mean, n_samples_mean)
                        if n_components_mean < 1:
                            raise ValueError(f"Not enough samples/features for PCA on mean_agg: shape={mean_agg.shape}")

                        # scale features (recommended) then PCA
                        scaler_mean = StandardScaler()
                        mean_agg_scaled = scaler_mean.fit_transform(mean_agg)
                        pca_mean = PCA(n_components=n_components_mean)
                        mean_2d = pca_mean.fit_transform(mean_agg_scaled)
                        print("mean 2d shape:", mean_2d.shape)

                        plt.figure(figsize=(8, 6))
                        plt.scatter(mean_2d[:, 0], mean_2d[:, 1] if n_components_mean > 1 else np.zeros_like(mean_2d[:, 0]), alpha=0.5)
                        plt.title('PCA of Mean Expert Index Vectors (aggregated)')
                        plt.xlabel('Principal Component 1')
                        plt.ylabel('Principal Component 2' if n_components_mean > 1 else '')
                        mean_fig = 'mean_expert_index_pca_agg.png'

                        plt.savefig(mean_fig)
                        plt.close()
                        print("mean fig saved")

                        # same for max_agg
                        n_samples_max, n_features_max = max_agg.shape
                        n_components_max = min(2, n_features_max, n_samples_max)
                        if n_components_max < 1:
                            raise ValueError(f"Not enough samples/features for PCA on max_agg: shape={max_agg.shape}")

                        scaler_max = StandardScaler()
                        max_agg_scaled = scaler_max.fit_transform(max_agg)
                        pca_max = PCA(n_components=n_components_max)
                        max_2d = pca_max.fit_transform(max_agg_scaled)

                        plt.figure(figsize=(8, 6))
                        plt.scatter(max_2d[:, 0], max_2d[:, 1] if n_components_max > 1 else np.zeros_like(max_2d[:, 0]), alpha=0.5)
                        plt.title('PCA of Max Expert Index Vectors (aggregated)')
                        plt.xlabel('Principal Component 1')
                        plt.ylabel('Principal Component 2' if n_components_max > 1 else '')
                        max_fig = 'max_expert_index_pca_agg.png'
                        plt.savefig(max_fig)
                        plt.close()
                        # --- PCA for full routing-weight vectors (concatenated per-layer experts) ---
                        try:
                            if len(full_accum) > 0:
                                full_agg = np.concatenate(full_accum, axis=0)  # (n_samples, n_layers * n_experts)
                                n_samples_full, n_features_full = full_agg.shape
                                n_components_full = min(2, n_features_full, n_samples_full)
                                if n_components_full >= 1:
                                    scaler_full = StandardScaler()
                                    full_agg_scaled = scaler_full.fit_transform(full_agg)
                                    pca_full = PCA(n_components=n_components_full)
                                    full_2d = pca_full.fit_transform(full_agg_scaled)

                                    plt.figure(figsize=(8, 6))
                                    plt.scatter(full_2d[:, 0], full_2d[:, 1] if n_components_full > 1 else np.zeros_like(full_2d[:, 0]), alpha=0.5)
                                    plt.title('PCA of Full Routing-Weight Vectors (aggregated)')
                                    plt.xlabel('Principal Component 1')
                                    plt.ylabel('Principal Component 2' if n_components_full > 1 else '')
                                    full_fig = 'full_expert_weights_pca_agg.png'
                                    plt.savefig(full_fig)
                                    plt.close()

                                    metrics_dict["test/full_expert_weights_pca"] = wandb.Image(full_fig)
                        except Exception as e:
                            print(f"[Warning] PCA on full routing-weight vectors skipped due to error: {e}")

                        # Log to wandb if available
                        metrics_dict["test/mean_expert_index_pca"] = wandb.Image(mean_fig)
                        metrics_dict["test/max_expert_index_pca"] = wandb.Image(max_fig)
                        accelerator.log(metrics_dict)
                        print("PCA aggregation analysis logged to wandb.")
                            
                       

                    except Exception as e:
                        # Do not crash the evaluation loop because of PCA issues
                        print(f"[Warning] PCA aggregation skipped due to error: {e}")

                # Clear accumulators so analysis can repeat later if desired
                mean_accum = []
                max_accum = []
                full_accum = []

            target_expert_mask = get_expert_mask(target_gate_logits, routing_strategy="top_k")

            #prediction
           
            #ignore padded tokens
            pad_mask = batch.get("attention_mask", None)
            if pad_mask is not None:
                pad_mask = pad_mask.view(-1, 1)
                pad_mask = (pad_mask != float("-inf")).squeeze(-1)
                target_expert_mask = target_expert_mask[:,pad_mask,:]

            #visualisation:


        # tokenizer = get_tokenizer(**cfg.tokenizer)
        # if i == 0:  # Only do this for the first batch

        #     visualisations_list = []
        #     # For 10 different random tokens in batch*seq_length, pick a token index,
        #     # extract the token _id from this,
        #     # decode it to the original token through the tokenizer,
        #     # and visualise the routing path.
            
        #     num_visualisations = 10
        #     input_ids_flat = batch["input_ids"].view(-1)
        #     seq_len = input_ids_flat.shape[0]
        #     # Randomly sample up to 10 unique token indices
        #     indices = np.random.choice(seq_len, min(num_visualisations, seq_len), replace=False)  
        #     for token_index in indices:
        #         token_id = input_ids_flat[token_index].item()
        #         try:
        #             token_str = tokenizer.decode([token_id])
        #         except Exception:
        #             token_str = str(token_id)
        #         _visualise_token_path(target_gate_logits, token_index, token_str, visualisations_list)
        #     metrics_dict["test/visualisations"] = visualisations_list

            # update progress bar per processed batch
            if accelerator.is_main_process and not cfg_predictor.trainer.disable_tqdm:
                pbar_test.update(1)
        pbar_test.close()




