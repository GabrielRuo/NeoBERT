from transformers import DataCollatorForLanguageModeling
import torch

def get_collatorCRAMMING(tokenizer, mlm_probability: float):
    """
    Create a DataCollator for language modeling (MLM or CLM).

    Args:
        tokenizer: PreTrainedTokenizerFast instance.
        mlm (bool): Whether to use masked language modeling.
        mlm_probability (float): Masking probability for MLM.

    Returns:
        A collator instance.
    """
    dtype = torch.float32

    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )
    def collate_fn(batch):
                batch = mlm_collator(batch)
                batch['attention_mask'] = None
                #batch["attention_mask"] = torch.where(batch["attention_mask"] == 1, float(0.0), float("-inf")).type(dtype)
                return batch
    
    return collate_fn
