"""Regression tests for Stage 1 packing and distillation settings."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from common.distill import distill_terms
from stage1.text_blocks import Packer


class FakeTokenizer:
    cls_token_id = 1
    sep_token_id = 2
    model_max_length = 0

    def __call__(self, texts, **kwargs):
        return {"input_ids": [[5 + (ord(char) % 90) for char in text] for text in texts]}


def test_packer_appends_separator_as_single_token():
    blocks = Packer(FakeTokenizer(), block_len=5).add(["a", "b", "c"])

    assert blocks == [[1, 12, 2, 13, 2], [1, 2, 14, 2, 2]]
    assert all(len(block) == 5 for block in blocks)


def test_removed_attention_loss_setting_is_rejected():
    with pytest.raises(ValueError, match="unknown loss setting"):
        distill_terms(None, None, None, None, None, {"attn": 0.0})