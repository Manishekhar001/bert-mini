"""Losses used by Stage 1 distillation and Stage 3 single-label classification."""
from __future__ import annotations

import torch.nn.functional as F


def kd_kl(student_logits, teacher_logits, temperature: float):
    """KL(teacher || student), averaged by batch and scaled by temperature squared."""
    import torch

    t = float(temperature)
    return F.kl_div(
        F.log_softmax(student_logits.float() / t, dim=-1),
        F.softmax(teacher_logits.float() / t, dim=-1),
        reduction="batchmean",
    ) * (t * t)


def classification_loss(logits, labels, class_weights=None):
    """Cross-entropy for one mutually-exclusive class label per report."""
    return F.cross_entropy(logits.float(), labels.long(), weight=class_weights)
