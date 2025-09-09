#load  pretrained model from modal
import os
import torch
from omegaconf import OmegaConf
from ..model import NeoBERTLMHead, NeoBERTConfig
from ..tokenizer import get_tokenizer
from ..pretraining.analysis import get_normalised_expert_usage_cost_per_sequence,get_entropy,get_mse_per_sequence

from ..dataset import get_dataset
from ..dataloader import get_dataloader
import numpy as np

def get_config(base_path):
    cfg_path = os.path.join(base_path,"config.yaml")
    cfg = OmegaConf.load(cfg_path)
    return cfg

def load_pretrained_models_modal(base_path,checkpoint):
    cfg = get_config(base_path)
    tokenizer = get_tokenizer(**cfg.tokenizer)
    neobert_model = NeoBERTLMHead(NeoBERTConfig(**cfg.model, **cfg.tokenizer, pad_token_id=tokenizer.pad_token_id))
    state_dict_path = os.path.join(base_path,"model_checkpoints",checkpoint,"state_dict.pt")
    neobert_state_dict = torch.load(state_dict_path, map_location="cpu")

    # Fix keys: strip "_orig_mod." if present
    new_state_dict = {}
    for k, v in neobert_state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod."):]] = v
        else:
            new_state_dict[k] = v

    neobert_model.load_state_dict(new_state_dict, strict=True)
    return neobert_model

def compute_correlations(base_path,checkpoint):
    cfg = get_config(base_path)
    model = load_pretrained_models_modal(base_path,checkpoint)
    tokenizer = get_tokenizer(**cfg.tokenizer)

    dtype_pad_mask = torch.float32
    # if cfg.trainer.mixed_precision== "fp16":
    #     dtype_pad_mask = torch.float16
    # elif cfg.trainer.mixed_precision== "bf16":
    #     dtype_pad_mask = torch.bfloat16

    test_dataset = get_dataset(cfg=cfg,hf_path="JeanKaddour/minipile",
    split= "test")

    dataloader = get_dataloader(test_dataset, tokenizer, dtype=dtype_pad_mask, **cfg.dataloader.train, **cfg.datacollator)

    mse_losses = []
    expert_usages = []
    i = 0
    for batch in dataloader:
        i += 1
        if i >= 500:
            break
        model_output = model(batch["input_ids"], batch.get("attention_mask", None), output_expert_usage_loss=True, output_router_logits=True)
        normalised_expert_usage_cost_per_seq = get_normalised_expert_usage_cost_per_sequence(model_output['router_logits'], batch.get("attention_mask", None), cfg)
        mse_loss_per_seq = get_mse_per_sequence(model_output['logits'], cfg, batch)

        # Convert to numpy arrays and flatten if needed
        mse_losses.extend(mse_loss_per_seq.detach().cpu().numpy().flatten())
        expert_usages.extend(normalised_expert_usage_cost_per_seq.detach().cpu().numpy().flatten())

    # Compute correlation
    mse_losses = np.array(mse_losses)
    expert_usages = np.array(expert_usages)
    correlation = np.corrcoef(mse_losses, expert_usages)[0, 1]
    print(f"Correlation between MSE loss per sequence and expert usage cost per sequence: {correlation}")

def compute_correlations_across_list_of_models(list_of_base_paths = [
"/runs/logs/checkpoints/mop_2025-08-27_15-27-10",
"/runs/logs/checkpoints/mop_2025-08-27_15-51-15",
"/runs/logs/checkpoints/mop_2025-08-27_15-54-23",
"/runs/logs/checkpoints/mop_2025-08-27_16-08-54",
"/runs/logs/checkpoints/mop_2025-08-27_16-52-12",
"/runs/logs/checkpoints/mop_2025-08-27_17-10-28",
"/runs/logs/checkpoints/mop_2025-08-27_17-43-18",
"/runs/logs/checkpoints/mop_2025-08-27_19-46-12"
], checkpoint="30000"):
    correlations = []
    for base_path in list_of_base_paths:
        print(f"Computing correlations for model at {base_path} with checkpoint {checkpoint}")
        cfg = get_config(base_path)
        print(f"alpha_start={cfg.model.loss.cost_based_loss_alpha_start},alpha_end={cfg.model.loss.cost_based_loss_alpha_end},schedule_tokens={cfg.model.loss.cost_based_loss_schedule_tokens}")
        correl = compute_correlations(base_path,checkpoint)
        print(f"Correlation for model: {correl}")
        correlations.append(correl)
    print(f"All correlations: {correlations}")

    
    












