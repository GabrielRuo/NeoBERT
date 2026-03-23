"""Pytest fixtures shared across all tests."""
import pytest


class MinimalTokenizer:
    """Minimal toy tokenizer for offline smoke tests.
    
    Avoids requiring HF downloads or cached tokenizers while providing
    just enough interface for training/inference tests.
    """
    
    def __init__(self):
        self.vocab_size = 1000
        self.pad_token_id = 0
        self.bos_token_id = 101
        self.eos_token_id = 102
        self.sep_token_id = 103
    
    def encode(self, text, **kwargs):
        """Return dummy token ids in range [1, vocab_size)."""
        if isinstance(text, str):
            return [hash(c) % (self.vocab_size - 1) + 1 for c in text[:10]]
        return text
    
    def decode(self, ids, **kwargs):
        """Return string representation of token ids."""
        return f"tok_{ids[0] if ids else 0}"
    
    def __call__(self, text, **kwargs):
        """Tokenize text to input_ids and attention_mask."""
        input_ids = self.encode(text, **kwargs)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }


@pytest.fixture
def minimal_tokenizer():
    """Fixture providing a minimal tokenizer for offline tests."""
    return MinimalTokenizer()
