from neobert.model.model import NeoBERTLMHead, NeoBERTConfig
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchEncoding
import warnings

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


# ajouter split_dataset_seed,test_size à la config, path to state dict
class SwiGLU(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim * 2, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, out_dim, bias=bias)

    def forward(self, x):
        x_proj = self.in_proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return self.out_proj(F.silu(x1) * x2)

def _resolve_mixed_precision(requested_mixed_precision: str) -> str:
    requested = str(requested_mixed_precision).lower()

    if requested == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        warnings.warn(
            "Requested bf16 but current CUDA device does not support it. Falling back to fp16."
        )
        return "fp16"

    if requested in {"fp16", "bf16"} and not torch.cuda.is_available():
        warnings.warn(
            f"Requested {requested} but CUDA is not available. Falling back to no mixed precision."
        )
        return "no"

    return requested

def _get_pretrained_neobert_model(cfg_pretrained, cfg_analysis, pad_token_id):
    neobert_model = NeoBERTLMHead(
        NeoBERTConfig(**cfg_pretrained.model, **cfg_pretrained.tokenizer, pad_token_id=pad_token_id)
    )
    state_dict_path = os.path.join(
        cfg_analysis.saved_model.base_path,
        "model_checkpoints",
        str(cfg_analysis.saved_model.checkpoint),
        "state_dict.pt",
    )
    neobert_state_dict = torch.load(state_dict_path, map_location="cpu")

    # Fix keys: strip "_orig_mod." if present
    new_state_dict = {}
    for k, v in neobert_state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod.") :]] = v
        else:
            new_state_dict[k] = v

    neobert_model.load_state_dict(new_state_dict, strict=True)
    return neobert_model
