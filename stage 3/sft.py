"""Stage 3: supervised fine-tuning for one mutually-exclusive class per report."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Trainer

from common.losses import classification_loss
from common.metrics import evaluate, logits_to_probs
from common.text_utils import pad_batch


class SftDataset(Dataset):
    def __init__(self, prepared_dataset):
        self.data = prepared_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        return {"input_ids": self.data.input_ids[index], "labels": self.data.labels[index]}


class SftCollator:
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict:
        ids, attention = pad_batch([f["input_ids"] for f in features], self.pad_id)
        labels = np.asarray([f["labels"] for f in features], dtype=np.int64)
        return {"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(attention),
                "labels": torch.from_numpy(labels)}


def class_weights(counts, mode: str | None):
    """Optional square-root inverse-frequency weights; each class remains a single choice."""
    if mode is None:
        return None
    if mode != "sqrt_inv":
        raise ValueError(f"class_weight must be null or 'sqrt_inv', got {mode!r}")
    if len(counts) < 2:
        raise ValueError("class weights need one count for each of at least two classes")
    frequency = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    weights = np.sqrt(frequency.sum() / (len(frequency) * frequency))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def make_compute_metrics():
    def compute(eval_pred):
        predictions = eval_pred.predictions
        logits = np.asarray(predictions[0] if isinstance(predictions, (tuple, list)) else predictions)
        metrics = evaluate(logits_to_probs(logits), np.asarray(eval_pred.label_ids))
        return {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "ece": metrics["ece"]}
    return compute


class ClassifierTrainer(Trainer):
    """Trainer with class-weighted cross-entropy and a separate learning rate for task pooling/output."""

    def __init__(self, *args, class_weights=None, head_lr: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.head_lr = head_lr
        try:
            self.model_accepts_loss_kwargs = False
        except AttributeError:
            pass

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = classification_loss(logits, labels, weights)
        return (loss, logits) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs["labels"]
        with torch.no_grad(), self.compute_loss_context_manager():
            loss, logits = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return loss.detach(), None, None
        return loss.detach(), logits.detach().float(), labels.detach()

    def create_optimizer(self, *args, **kwargs):
        if self.optimizer is not None:
            return self.optimizer
        decay, no_decay, task_params = [], [], []
        model = self.model
        for name, parameter in model.encoder.named_parameters():
            if parameter.requires_grad:
                (no_decay if parameter.ndim < 2 or "LayerNorm" in name or "bias" in name else decay).append(parameter)
        task_params.extend(p for p in model.classifier.parameters() if p.requires_grad)
        if model.pooler is not None:
            task_params.extend(p for p in model.pooler.parameters() if p.requires_grad)
        groups = []
        if decay:
            groups.append({"params": decay, "lr": self.args.learning_rate, "weight_decay": self.args.weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": self.args.learning_rate, "weight_decay": 0.0})
        if task_params:
            groups.append({"params": task_params, "lr": self.head_lr or self.args.learning_rate,
                           "weight_decay": 0.0})
        self.optimizer = torch.optim.AdamW(
            groups, betas=(self.args.adam_beta1, self.args.adam_beta2), eps=self.args.adam_epsilon
        )
        return self.optimizer


@torch.no_grad()
def predict_logits(model, dataset, device, batch_size: int = 64, amp: bool = False) -> np.ndarray:
    """Get one [N, classes] matrix in original row order (fp32 for calibration/serving parity)."""
    from common.utils import autocast_ctx

    model.eval()
    order = np.argsort([len(ids) for ids in dataset.input_ids], kind="stable")
    logits_out = np.zeros((len(dataset), dataset.schema.n_classes), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        batch = dataset.batch(indices)
        amp_context = autocast_ctx(device) if amp else torch.autocast(device.type, enabled=False)
        with amp_context:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        logits_out[indices] = logits.float().cpu().numpy()
    return logits_out
