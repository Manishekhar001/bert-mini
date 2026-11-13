"""Text helpers with no torch dependency: tokenization, head+tail truncation and padding."""
from __future__ import annotations

import numpy as np


def head_tail_ids(ids, cls_id: int, sep_id: int, max_len: int = 256, head_tokens: int = 192) -> list:
    """[CLS] + (first head_tokens + last tokens) + [SEP]; keeps the opening AND the outcome of long reports."""
    if max_len < 2:
        raise ValueError(f"max_len must leave room for [CLS] and [SEP], got {max_len}")
    if head_tokens < 0:
        raise ValueError(f"head_tokens cannot be negative, got {head_tokens}")
    body = max_len - 2
    head_tokens = min(head_tokens, body)
    ids = list(ids)
    if len(ids) > body:
        tail = body - head_tokens
        ids = ids[:head_tokens] + (ids[-tail:] if tail > 0 else [])
    return [cls_id] + ids + [sep_id]


def tokenize_texts(tokenizer, texts, max_len: int = 256, head_tokens: int = 192, chunk: int = 2000) -> list:
    """Tokenize with the fast tokenizer, then apply head+tail truncation. Returns list of id lists."""
    if chunk < 1:
        raise ValueError(f"chunk must be positive, got {chunk}")
    out = []
    texts = list(texts)
    for i in range(0, len(texts), chunk):
        enc = tokenizer(texts[i:i + chunk], add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        out.extend(head_tail_ids(e, tokenizer.cls_token_id, tokenizer.sep_token_id, max_len, head_tokens) for e in enc)
    return out


def pad_batch(seqs, pad_id: int = 0):
    """Pad a list of id lists to the longest one. Returns (input_ids, attention_mask) as int64 arrays."""
    if not seqs:
        raise ValueError("pad_batch needs at least one token sequence")
    length = max(len(s) for s in seqs)
    ids = np.full((len(seqs), length), pad_id, dtype=np.int64)
    att = np.zeros((len(seqs), length), dtype=np.int64)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
        att[i, :len(s)] = 1
    return ids, att
