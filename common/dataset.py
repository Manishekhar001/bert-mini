"""Tokenized view of the prepared one-report, one-class ASRS dataset."""
from __future__ import annotations

import numpy as np

from .text_utils import pad_batch, tokenize_texts


class AsrsDataset:
    def __init__(self, rows, schema, tokenizer, max_len: int = 256, head_tokens: int = 192):
        self.schema = schema
        self.ids = [str(row["id"]) for row in rows]
        self.input_ids = tokenize_texts(
            tokenizer, [row["text"] for row in rows], max_len=max_len, head_tokens=head_tokens
        )
        self.pad_id = tokenizer.pad_token_id or 0
        self.labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        if np.any(self.labels < 0) or np.any(self.labels >= schema.n_classes):
            raise ValueError("A prepared label is outside the class-name mapping in schema_resolved.json")

    def __len__(self) -> int:
        return len(self.input_ids)

    def batch(self, indices) -> dict:
        import torch

        indices = np.asarray(indices)
        ids, attention = pad_batch([self.input_ids[i] for i in indices], self.pad_id)
        return {
            "input_ids": torch.from_numpy(ids),
            "attention_mask": torch.from_numpy(attention),
            "labels": torch.from_numpy(self.labels[indices]),
        }
