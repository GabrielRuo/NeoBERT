import os
from transformers import PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download

from datasets import load_dataset,load_from_disk, Dataset
from typing import Optional, Any
import logging

from pathlib import Path


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)#not sure why


# def get_dataset(
#     dataset_name: str = "JonasGeiping/the_pile_WordPiecex32768_8eb2d0ea9da707676c81314c4ea04507",
#     split: str = "train",
#     num_samples: Optional[int] = None,
#     streaming: bool = False,
# ):
#     """
#     Load and return a Hugging Face dataset, optionally truncated to `num_samples`.

#     Args:
#         dataset_name (str): The HF dataset identifier.
#         split (str): Dataset split to load (e.g. "train").
#         num_samples (int, optional): If provided, return only this many samples.
#         streaming (bool): If True, use streaming mode (for very large datasets).

#     Returns:
#         Dataset: A Hugging Face dataset, formatted for PyTorch.
#     """
#     logger.info(f"Loading dataset: {dataset_name} (split={split}, streaming={streaming})")

#     # dataset = load_dataset(dataset_name, split=split, streaming=streaming)

#     # if not streaming:
#     #     if num_samples is not None:
#     #         logger.info(f"Selecting {num_samples} samples from dataset")
#     #         dataset = dataset.select(range(min(num_samples, len(dataset))))

#     #     # Format for PyTorch
#     #     dataset.set_format(type="torch", columns=["input_ids"])
#     # else:
#     #     # If streaming, dataset is an iterable — can't use select or set_format
#     #     logger.warning("Dataset is in streaming mode — 'num_samples' and formatting will be ignored")

#     dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]") #must have streaming = False

#     return dataset

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


def get_datasetCRAMMING(
    hf_path: str = "JonasGeiping/the_pile_WordPiecex32768_8eb2d0ea9da707676c81314c4ea04507",
    split: str = "train",
    num_samples: Optional[int] = None,
    streaming: bool = False,
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

    if num_samples is None:
        logger.info(f"loading full dataset from {hf_path} ...")
        dataset_path = get_dataset_path(hf_path, num_samples)
    else: 
        dataset_path = get_dataset_path(hf_path, num_samples)


    if dataset_path.exists():
        logger.info(f"Loading cached dataset from: {dataset_path}")
        dataset = load_from_disk(dataset_path)
    else:
        if num_samples is None:
            dataset = load_dataset(hf_path, split=split, streaming=False)
        else:
            logger.info(f"Downloading and caching {hf_path} with {num_samples} samples...")
            dataset = load_dataset(hf_path, split=f"{split}[:{num_samples}]", streaming=False)
        dataset.set_format(type="torch", columns=["input_ids"])
        dataset.save_to_disk(dataset_path)

    return dataset

def get_tokenizerCRAMMING(tokenizer_parent_dir=None):
    """
    Load or download the tokenizer used for the JonasGeiping WordPiece dataset.

    Args:
        tokenizer_dir (str): Absolute or relative path to tokenizer folder.
                             If None, defaults to "../tokenizer" relative to this file.
    """
    # Resolve default path relative to this file
    if tokenizer_parent_dir is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))  
        tokenizer_parent_dir = os.path.abspath(os.path.join(this_dir, ".."))

    # Download files if needed
    for filename in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        hf_hub_download(
        repo_id="JonasGeiping/the_pile_WordPiecex32768_8eb2d0ea9da707676c81314c4ea04507",
        filename=f"tokenizer/{filename}",
        repo_type="dataset",
        local_dir=tokenizer_parent_dir,
        )
    print(tokenizer_parent_dir)

    # Load tokenizer
    tokenizer_dir = os.path.abspath(os.path.join(tokenizer_parent_dir, "tokenizer"))
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    return tokenizer



