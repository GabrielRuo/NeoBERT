from transformers import DataCollatorForLanguageModeling


def get_collator_pile(tokenizer, mlm_probability: float):
    """
    Create a DataCollator for masked language modeling for the pile dataset.

    Args:
        tokenizer: PreTrainedTokenizerFast instance.
        mlm_probability (float): Masking probability for MLM.

    Returns:
        A collator callable.
    """
    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
    )

    def collate_fn(batch):
        batch = mlm_collator(batch)
        batch["attention_mask"] = None
        return batch

    return collate_fn
