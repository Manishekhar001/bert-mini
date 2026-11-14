"""Predict one class for each report, with optional review and out-of-domain flags."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common.schema import Schema
from common.text_utils import pad_batch, tokenize_texts
from common.utils import read_json
from stage4.calibration import calibrated_probs, domain_score


def torch_logits_for_texts(model, tokenizer, texts, device, max_len: int = 256, head_tokens: int = 192,
                           batch_size: int = 32) -> np.ndarray:
    """Return one [number of reports, number of classes] logit matrix."""
    import torch

    texts = list(texts)
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if not texts:
        return np.zeros((0, model.num_classes), dtype=np.float32)
    sequences = tokenize_texts(tokenizer, texts, max_len, head_tokens)
    output = np.zeros((len(sequences), model.num_classes), dtype=np.float32)
    order = np.argsort([len(sequence) for sequence in sequences], kind="stable")
    model.eval()
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            ids, attention = pad_batch([sequences[i] for i in indices], tokenizer.pad_token_id or 0)
            logits = model(torch.from_numpy(ids).to(device), torch.from_numpy(attention).to(device))
            output[indices] = logits.float().cpu().numpy()
    return output


class Predictor:
    """Load the package emitted by stage4/calibrate.py."""

    BACKENDS = {"torch": None, "onnx": "model.onnx", "onnx_int8": "model.int8.onnx"}

    def __init__(self, model_dir, backend: str = "torch", max_len: int = 256, head_tokens: int = 192,
                 batch_size: int = 32, num_threads: int | None = None):
        from transformers import BertTokenizerFast

        self.dir = Path(model_dir)
        self.schema = Schema.load(self.dir / "schema_resolved.json")
        self.cal = read_json(self.dir / "calibration.json")
        self.tokenizer = BertTokenizerFast.from_pretrained(str(self.dir / "tokenizer"))
        data_cfg = self.cal.get("data_cfg", {})
        self.max_len = data_cfg.get("max_len", max_len)
        self.head_tokens = data_cfg.get("head_tokens", head_tokens)
        self.batch_size, self.backend = batch_size, backend
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if backend == "torch":
            import torch
            from common.model import load_classifier

            if num_threads:
                torch.set_num_threads(num_threads)
            self.model = load_classifier(self.dir / "model.pt").eval()
            if self.model.num_classes != self.schema.n_classes:
                raise ValueError("model output size does not match the packaged class-name mapping")
        elif backend in ("onnx", "onnx_int8"):
            import onnxruntime as ort

            options = ort.SessionOptions()
            if num_threads:
                options.intra_op_num_threads = num_threads
            self.session = ort.InferenceSession(str(self.dir / self.BACKENDS[backend]), options,
                                                providers=["CPUExecutionProvider"])
        else:
            raise ValueError(f"backend must be one of {list(self.BACKENDS)}")

    def logits(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if self.backend == "torch":
            import torch

            return torch_logits_for_texts(self.model, self.tokenizer, texts, torch.device("cpu"), self.max_len,
                                          self.head_tokens, self.batch_size)
        sequences = tokenize_texts(self.tokenizer, texts, self.max_len, self.head_tokens)
        output = np.zeros((len(sequences), self.schema.n_classes), dtype=np.float32)
        order = np.argsort([len(sequence) for sequence in sequences], kind="stable")
        for start in range(0, len(order), self.batch_size):
            indices = order[start:start + self.batch_size]
            ids, attention = pad_batch([sequences[i] for i in indices], self.tokenizer.pad_token_id or 0)
            result = self.session.run(["logits"], {"input_ids": ids, "attention_mask": attention})[0]
            output[indices] = result
        return output

    def predict(self, texts) -> list[dict]:
        texts = [texts] if isinstance(texts, str) else list(texts)
        if not texts:
            return []
        logits = self.logits(texts)
        probs = calibrated_probs(logits, self.cal.get("temperature", 1.0))
        ood = self.cal.get("ood")
        scores = domain_score(ood["score"], logits, probs) if ood else None
        results = []
        for i, probabilities in enumerate(probs):
            class_id = int(probabilities.argmax())
            confidence = float(probabilities[class_id])
            reasons = []
            cutoff = self.cal.get("confidence_threshold")
            if cutoff is None or confidence < float(cutoff):
                reasons.append("low_confidence")
            is_ood = bool(ood and scores[i] < float(ood["threshold"]))
            if is_ood:
                reasons.append("out_of_domain")
            results.append({
                "label": self.schema.classes[class_id],
                "label_id": class_id,
                "confidence": confidence,
                "probabilities": {name: float(probability)
                                  for name, probability in zip(self.schema.classes, probabilities)},
                "needs_review": bool(reasons),
                "review_reasons": reasons,
                "out_of_domain": is_ood,
            })
        return results
