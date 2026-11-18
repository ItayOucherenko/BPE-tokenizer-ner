"""
Abstract tokenizer interface used by the BPE tokenizer and the NER training script.

Subclasses implement train(), encode() and decode(). The class also provides batch
helpers, vocabulary size, and pickle-based save/load.
"""

import os
import pickle
from abc import ABC, abstractmethod
from typing import List


class BaseTokenizer(ABC):
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3}
        for token, token_id in self.special_tokens.items():
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

    @abstractmethod
    def train(self, texts: List[str]) -> None:
        """Learn the vocabulary from a list of training texts."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert a string into a list of token IDs."""

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs back into a string."""

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts]

    def decode_batch(self, batch_token_ids: List[List[int]]) -> List[str]:
        return [self.decode(token_ids) for token_ids in batch_token_ids]

    def get_vocab_size(self) -> int:
        return len(self.token_to_id)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaseTokenizer":
        with open(path, "rb") as f:
            return pickle.load(f)
