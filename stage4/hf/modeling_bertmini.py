"""bert-mini ASRS report classifier in Hugging Face format (self-contained; needs only torch and transformers).

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(repo, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model.classify(["narrative text ..."], tokenizer)

`classify` repeats the project's own pipeline: head + tail truncation, the per-class prior correction, temperature
scaling, and the review flag. Plain `model(input_ids, attention_mask).logits` returns raw scores.
"""
import torch
import torch.nn as nn
from transformers import BertModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .configuration_bertmini import BertMiniConfig


class AttentivePooler(nn.Module):
    """Attention pooling: learn one report vector by weighting its token vectors."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1, bias=False))

    def forward(self, seq, attention_mask):
        scores = self.proj(seq).squeeze(-1).masked_fill(attention_mask == 0, -1e4)
        return (seq * torch.softmax(scores, dim=-1).unsqueeze(-1)).sum(dim=1)


def head_tail_ids(ids, cls_id, sep_id, max_len, head_tokens):
    """[CLS] + (first head_tokens + last tokens) + [SEP]; keeps the opening AND the outcome of long reports."""
    body = max_len - 2
    head_tokens = min(head_tokens, body)
    ids = list(ids)
    if len(ids) > body:
        tail = body - head_tokens
        ids = ids[:head_tokens] + (ids[-tail:] if tail > 0 else [])
    return [cls_id] + ids + [sep_id]


class BertMiniForSequenceClassification(PreTrainedModel):
    config_class = BertMiniConfig
    base_model_prefix = "encoder"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.encoder = BertModel(config, add_pooling_layer=False)
        self.pooler = AttentivePooler(config.hidden_size) if config.pooling == "attn" else None
        self.drop = nn.Dropout(config.classifier_dropout if config.classifier_dropout is not None else 0.1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        seq = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids).last_hidden_state
        if self.config.pooling == "attn":
            pooled = self.pooler(seq, attention_mask)
        elif self.config.pooling == "cls":
            pooled = seq[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(seq.dtype)
            pooled = (seq * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        logits = self.classifier(self.drop(pooled))
        loss = nn.functional.cross_entropy(logits.float(), labels.long()) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def classify(self, texts, tokenizer, batch_size: int = 16):
        """One dict per report: label, calibrated confidence, all probabilities, review flags."""
        texts = [texts] if isinstance(texts, str) else list(texts)
        cfg = self.config
        max_len = int(cfg.report_max_len)
        head_tokens = int(cfg.report_head_tokens)
        device = next(self.parameters()).device
        self.eval()
        encoded = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
        sequences = [head_tail_ids(ids, tokenizer.cls_token_id, tokenizer.sep_token_id, max_len, head_tokens)
                     for ids in encoded]
        bias = torch.tensor(cfg.class_bias, dtype=torch.float64) if cfg.class_bias else None
        temperature = float(cfg.calibration_temperature)
        cutoff = cfg.confidence_threshold
        pad_id = tokenizer.pad_token_id or 0
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        results = [None] * len(sequences)
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            longest = max(len(sequences[i]) for i in index)
            ids = torch.full((len(index), longest), pad_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for row, i in enumerate(index):
                ids[row, :len(sequences[i])] = torch.tensor(sequences[i])
                mask[row, :len(sequences[i])] = 1
            logits = self(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.double().cpu()
            if bias is not None:
                logits = logits + bias
            probs = torch.softmax(logits / temperature, dim=-1)
            for row, i in enumerate(index):
                p = probs[row]
                best = int(p.argmax())
                confidence = float(p[best])
                reasons = []
                if cutoff is None or confidence < float(cutoff):
                    reasons.append("low_confidence")
                results[i] = {
                    "label": cfg.id2label[best], "label_id": best, "confidence": confidence,
                    "probabilities": {cfg.id2label[j]: float(p[j]) for j in range(len(p))},
                    "needs_review": bool(reasons), "review_reasons": reasons,
                }
        return results
