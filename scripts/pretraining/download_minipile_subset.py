import argparse
from itertools import islice
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


DEFAULT_DATASET = "JeanKaddour/minipile"


def build_subset(split: str, num_samples: int, hf_dataset: str) -> Dataset:
    stream = load_dataset(hf_dataset, split=split, streaming=True)
    records = list(islice(stream, num_samples))
    if not records:
        raise ValueError(
            f"No records were downloaded for split '{split}'. Check dataset id and split name."
        )
    return Dataset.from_list(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a tiny subset of MiniPile and save it locally for fast tests."
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=100,
        help="Number of train samples to keep.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=20,
        help="Number of test samples to keep.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/minipile_tiny",
        help="Output directory inside the repository.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / args.output_dir
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    train_ds = build_subset("train", args.train_samples, args.hf_dataset)
    test_ds = build_subset("test", args.test_samples, args.hf_dataset)

    subset = DatasetDict({"train": train_ds, "test": test_ds})
    subset.save_to_disk(str(output_dir))

    print(f"Saved tiny dataset to: {output_dir}")
    print(f"train rows: {len(train_ds)}")
    print(f"test rows: {len(test_ds)}")


if __name__ == "__main__":
    main()
