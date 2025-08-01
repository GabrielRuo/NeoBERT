from transformers import DataCollatorForLanguageModeling


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
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )
