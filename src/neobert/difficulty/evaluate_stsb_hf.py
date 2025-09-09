from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
import numpy as np
from scipy.stats import pearsonr

def main():
    # Load model and tokenizer
    model_name = "textattack/bert-base-uncased-STS-B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Load validation set
    stsb = load_dataset("glue", "stsb")
    val_set = stsb["validation"]

    preds = []
    golds = []

    for example in val_set:
        inputs = tokenizer(
            example["sentence1"],
            example["sentence2"],
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.logits.item()
        preds.append(pred)
        golds.append(example["label"])

    preds = np.array(preds)
    golds = np.array(golds)
    mse = np.mean((preds - golds) ** 2)
    pearson = pearsonr(preds, golds)[0]

    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation Pearson correlation: {pearson:.4f}")

if __name__ == "__main__":
    main()
