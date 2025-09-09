import os
from transformers import PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download

from datasets import load_dataset,load_from_disk, Dataset
from typing import Optional, Any
import logging

from pathlib import Path

from neobert.tokenizer import get_tokenizer, tokenize


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)#not sure why


def get_dataset_path(dataset_name: str, num_samples: Optional[int]) -> Path:
    """Construct a unique path for caching the dataset."""
    base_dir = Path("/data") if Path("/data").exists() else Path.home()
    cache_root = base_dir / ".pathways_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    dataset_key = ''.join(c for c in dataset_name if c.isalnum()).lower()
    if num_samples is None:
        return cache_root / f"{dataset_key}"
    else:
        return cache_root / f"{dataset_key}_{num_samples}"


def get_dataset(
    cfg,
    hf_path: str = "https://huggingface.co/datasets/JeanKaddour/minipile",
    split: str = "train",
    num_samples: Optional[int] = None,
    streaming: bool = False,
    subset: Optional[str] = None,  # <-- add this argument
) -> Dataset:
    """
    Load and cache a Hugging Face dataset for efficient reuse.
    
    Args:
        dataset_name (str): Dataset identifier on Hugging Face.
        split (str): Dataset split, usually "train".
        num_samples (int, optional): Number of samples to load.
        streaming (bool): If True, use streaming mode (no caching).

    Returns:
        Dataset: The loaded dataset, formatted for PyTorch.
    """
    #assert not streaming, "Streaming mode is not supported with caching"

    dataset_name = hf_path + (f"_{subset}" if subset else "") + "_" + split
    if num_samples is None:
        logger.info(f"loading full dataset from {hf_path} ...")
        dataset_path = get_dataset_path(dataset_name, num_samples)
    else: 
        dataset_path = get_dataset_path(dataset_name, num_samples)


    if dataset_path.exists():
        logger.info(f"Loading cached dataset from: {dataset_path}")
        dataset = load_from_disk(dataset_path)
    else:
        if num_samples is None:
            if subset:
                dataset = load_dataset(hf_path, subset, split=split, streaming=False)
            else:
                dataset = load_dataset(hf_path, split=split, streaming=False)
        else:
            logger.info(f"Downloading and caching {hf_path} with {num_samples} samples...")
            if subset:
                dataset = load_dataset(hf_path, subset, split=f"{split}[:{num_samples}]", streaming=False)
            else:
                dataset = load_dataset(hf_path, split=f"{split}[:{num_samples}]", streaming=False)
        #tokenize
        tokenizer = get_tokenizer(**cfg.tokenizer)
        dataset = tokenize(dataset, tokenizer, column_name=cfg.dataset.column, **cfg.tokenizer)
        dataset.save_to_disk(dataset_path)

    return dataset
