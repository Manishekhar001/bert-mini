"""BERT-mini student, single-output report classifier and checkpoint helpers."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig, BertForMaskedLM, BertModel


def student_config(s: dict, attn_impl: str | None = None) -> BertConfig:
    """BERT-mini by default: 4 layers, hidden 256, 4 heads, FFN 1024, 256 positions (~11.1M parameters)."""
    cfg = BertConfig(
        vocab_size=s.get("vocab_size", 30522),
        hidden_size=s["hidden_size"],
        num_hidden_layers=s["num_layers"],
        num_attention_heads=s["num_heads"],
        intermediate_size=s["intermediate_size"],
        max_position_embeddings=s["max_positions"],
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    if attn_impl:
        cfg._attn_implementation = attn_impl        # 'eager' is required to read attention maps
    return cfg


def build_student_mlm(s: dict, attn_impl: str | None = None) -> BertForMaskedLM:
    return BertForMaskedLM(student_config(s, attn_impl))      # random weights: "from scratch"


def load_student_mlm(path_or_name, attn_impl: str | None = None) -> BertForMaskedLM:
    """Load a student saved with save_pretrained (a folder) or any BertForMaskedLM on the Hub. Always fp32."""
    kw = {"attn_implementation": attn_impl} if attn_impl else {}
    return BertForMaskedLM.from_pretrained(str(path_or_name), **kw).float()


def extend_positions(model, new_max: int, mode: str = "smooth"):
    """Grow the learned position table to `new_max` positions IN PLACE.

    mode='smooth': positions 0..old-1 are unchanged. New positions start as copies
                   perturbed with small Gaussian noise to break exact duplicate symmetry.
    mode='tile'  : exact copy (legacy).
    """
    bert = model.bert if hasattr(model, "bert") else model
    emb = bert.embeddings
    old = emb.position_embeddings.weight.data
    n_old, dim = old.shape
    if new_max <= n_old:
        return model
    new_weight = torch.empty(new_max, dim, device=old.device, dtype=old.dtype)
    new_weight[:n_old] = old
    reps = math.ceil(new_max / n_old)
    tiled = old.repeat(reps, 1)[:new_max]

    if mode == "smooth":
        scale = float(old.std().item() if hasattr(old, "std") else 0.02) * 0.02
        noise = torch.randn_like(tiled[n_old:]) * scale
        new_weight[n_old:] = tiled[n_old:] + noise
    else:
        new_weight[n_old:] = tiled[n_old:]

    new_emb = nn.Embedding(new_max, dim).to(device=old.device, dtype=old.dtype)
    new_emb.weight.data.copy_(new_weight)
    emb.position_embeddings = new_emb
    emb.register_buffer("position_ids", torch.arange(new_max, device=old.device).unsqueeze(0), persistent=False)
    if hasattr(emb, "token_type_ids"):
        emb.register_buffer("token_type_ids", torch.zeros(1, new_max, dtype=torch.long, device=old.device),
                            persistent=False)
    bert.config.max_position_embeddings = new_max
    model.config.max_position_embeddings = new_max
    return model


def build_encoder(init: str, student_cfg: dict | None = None, max_positions: int | None = None):
    """Load a BERT encoder for the classifier from random weights, a local folder, or ``hf:<model id>``.

    The classifier checkpoint and LoRA implementation currently require BERT architecture. Reject other
    Hugging Face encoder families here, before they fail during training or when loading the saved classifier.
    ``max_positions`` extends the position table when the checkpoint has fewer positions than requested.
    """
    if init == "random":
        cfg = dict(student_cfg or {})
        if max_positions:
            cfg["max_positions"] = max(max_positions, cfg.get("max_positions", 0))
        return BertModel(student_config(cfg), add_pooling_layer=False)
    name = init[3:] if init.startswith("hf:") else init
    try:
        enc = AutoModel.from_pretrained(name, add_pooling_layer=False)
    except TypeError:                          # e.g. DistilBERT has no pooler argument
        enc = AutoModel.from_pretrained(name)
    if not isinstance(enc, BertModel):
        model_type = getattr(enc.config, "model_type", type(enc).__name__)
        raise ValueError(
            f"Stage 3 currently supports BERT encoders only; {name!r} loaded as {model_type!r}. "
            "Choose a BERT checkpoint or add matching save/load support first."
        )
    enc = enc.float()                          # newer transformers may load checkpoints in their stored dtype
    if max_positions and getattr(enc.config, "max_position_embeddings", 0) < max_positions:
        extend_positions(enc, max_positions)
    return enc


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) layer wrapping a base nn.Linear.

    h = W_0 * x + (alpha / r) * (x @ A.T @ B.T)
    Frozen W_0, trainable A and B. Merges back to base nn.Linear with zero inference overhead.
    """

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if r > 0:
            in_dim = base_linear.in_features
            out_dim = base_linear.out_features
            self.lora_A = nn.Parameter(torch.empty(r, in_dim, device=base_linear.weight.device, dtype=base_linear.weight.dtype))
            self.lora_B = nn.Parameter(torch.zeros(out_dim, r, device=base_linear.weight.device, dtype=base_linear.weight.dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r > 0 and self.lora_A is not None:
            lora_out = (self.drop(x) @ self.lora_A.T) @ self.lora_B.T * self.scaling
            out = out + lora_out
        return out

    def merge_to_base(self) -> nn.Linear:
        """Fold LoRA weights into base.weight and return the bare base nn.Linear."""
        if self.r > 0 and self.lora_A is not None:
            delta = (self.lora_B.data @ self.lora_A.data) * self.scaling
            self.base.weight.data.add_(delta.to(self.base.weight.device, dtype=self.base.weight.dtype))
            self.r = 0
            self.lora_A = None
            self.lora_B = None
        return self.base


def apply_lora_to_mlm(model: BertForMaskedLM, r: int = 8, alpha: float = 16.0,
                      dropout: float = 0.05, target_modules: tuple = ("query", "value")) -> BertForMaskedLM:
    """Freeze a BertForMaskedLM, add LoRA to attention projections, and train its position table.

    This is used for Stage 2 domain-adaptive MLM. All original weights are frozen except the position
    embeddings; the injected LoRA A/B matrices and the position embeddings are trainable. Keeping the
    position table trainable lets the model adapt the extended 512-position context as well as the domain.
    """
    if r < 1:
        raise ValueError(f"LoRA rank must be positive, got {r}")
    if not hasattr(model, "bert") or not hasattr(model.bert, "encoder"):
        raise TypeError("Stage 2 LoRA expects a BertForMaskedLM model with a BERT encoder")
    target_modules = tuple(target_modules)
    if not target_modules:
        raise ValueError("At least one LoRA target must be configured")
    if len(set(target_modules)) != len(target_modules):
        raise ValueError(f"LoRA targets contain duplicates: {target_modules}")

    plans = []
    for layer in model.bert.encoder.layer:
        attn = layer.attention.self
        for target in target_modules:
            base = getattr(attn, target, None)
            if isinstance(base, LoRALinear):
                raise ValueError(f"LoRA is already injected into attention.{target}")
            if not isinstance(base, nn.Linear):
                raise ValueError(f"BERT attention target {target!r} is not a Linear layer")
            plans.append((attn, target, base))

    if not plans:
        raise ValueError("No LoRA targets were injected; check target_modules")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for attn, target, base in plans:
        setattr(attn, target, LoRALinear(base, r=r, alpha=alpha, dropout=dropout))
    model.bert.embeddings.position_embeddings.weight.requires_grad_(True)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    targets = ", ".join(target_modules)
    print(f"[lora] Stage 2: injected into {len(plans)} {targets} projections; trainable {trainable:,}/{total:,} "
          f"parameters (LoRA + position embeddings)")
    return model


def merge_lora_mlm(model: BertForMaskedLM) -> BertForMaskedLM:
    """Merge Stage 2 LoRA deltas into their base projections for a standard save_pretrained artifact."""
    if not hasattr(model, "bert") or not hasattr(model.bert, "encoder"):
        raise TypeError("Stage 2 LoRA merge expects a BertForMaskedLM model with a BERT encoder")
    count = 0
    for layer in model.bert.encoder.layer:
        attn = layer.attention.self
        for name in list(attn._modules):
            module = getattr(attn, name)
            if isinstance(module, LoRALinear):
                setattr(attn, name, module.merge_to_base())
                count += 1
    if count:
        print(f"[lora] merged {count} Stage 2 adapters into standard BERT linear layers")
    return model


class AttentivePooler(nn.Module):
    """Attention pooling: learn one report vector by weighting its token vectors."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, seq: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.proj(seq).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
        return (seq * weights).sum(dim=1)  # [B, hidden]


class ReportClassifier(nn.Module):
    """One shared encoder, one pooled report vector, and one multiclass output layer."""

    def __init__(self, encoder: nn.Module, num_classes: int, pooling: str = "attn", dropout: float = 0.1):
        super().__init__()
        if pooling not in {"attn", "mean", "cls"}:
            raise ValueError(f"pooling must be attn, mean or cls; got {pooling!r}")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.encoder = encoder
        self.pooling = pooling
        self.drop = nn.Dropout(dropout)
        hidden = encoder.config.hidden_size if hasattr(encoder.config, "hidden_size") else encoder.config.dim
        self.pooler = AttentivePooler(hidden) if pooling == "attn" else None
        self.classifier = nn.Linear(hidden, num_classes)

    @property
    def num_classes(self) -> int:
        return self.classifier.out_features

    def forward(self, input_ids, attention_mask):
        seq = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "attn":
            pooled = self.pooler(seq, attention_mask)
        elif self.pooling == "cls":
            pooled = seq[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(seq.dtype)
            pooled = (seq * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.classifier(self.drop(pooled))

    def apply_lora(self, r: int = 8, alpha: float = 16.0, dropout: float = 0.05,
                   target_modules: tuple = ("query", "value")):
        """Freeze encoder weights and inject LoRALinear into self-attention targets."""
        if r < 1:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        target_modules = tuple(target_modules)
        if not target_modules:
            raise ValueError("At least one LoRA target must be configured")
        if len(set(target_modules)) != len(target_modules):
            raise ValueError(f"LoRA targets contain duplicates: {target_modules}")

        enc = self.encoder.encoder if hasattr(self.encoder, "encoder") else self.encoder
        if not hasattr(enc, "layer"):
            raise TypeError("LoRA injection currently supports BERT encoders exposing encoder.layer")
        plans = []
        for layer in enc.layer:
            attn = layer.attention.self
            for target in target_modules:
                base = getattr(attn, target, None)
                if isinstance(base, LoRALinear):
                    raise ValueError(f"LoRA is already injected into attention.{target}")
                if not isinstance(base, nn.Linear):
                    raise ValueError(f"BERT attention target {target!r} is not a Linear layer")
                plans.append((attn, target, base))
        if not plans:
            raise ValueError("No LoRA targets were injected; check target_modules")

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for attn, target, base in plans:
            setattr(attn, target, LoRALinear(base, r=r, alpha=alpha, dropout=dropout))
        targets = ", ".join(target_modules)
        print(f"[lora] injected LoRA (r={r}, alpha={alpha}) into {len(plans)} {targets} projections; encoder frozen")
        return self

    def merge_lora(self):
        """Fold all LoRA weights back into the standard linear layers (zero inference overhead)."""
        count = 0
        enc = self.encoder.encoder if hasattr(self.encoder, "encoder") else self.encoder
        for layer in enc.layer:
            attn = layer.attention.self
            for name in list(attn._modules.keys()):
                mod = getattr(attn, name)
                if isinstance(mod, LoRALinear):
                    setattr(attn, name, mod.merge_to_base())
                    count += 1
        if count > 0:
            print(f"[lora] merged {count} LoRA layers into base model")
        return self


def save_classifier(path, model: ReportClassifier) -> None:
    if hasattr(model, "merge_lora"):
        model.merge_lora()
    torch.save(
        {
            "bert_config": model.encoder.config.to_dict(),
            "state_dict": model.state_dict(),
            "pooling": model.pooling,
            "num_classes": model.num_classes,
        },
        path,
    )


def load_classifier(path, map_location: str = "cpu", attn_impl: str | None = None) -> ReportClassifier:
    """Reload a student classifier (BERT encoder without pooler) saved by save_classifier."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = BertConfig.from_dict(ck["bert_config"])
    if attn_impl:
        cfg._attn_implementation = attn_impl     # 'eager' is the safest choice for ONNX export
    model = ReportClassifier(BertModel(cfg, add_pooling_layer=False), ck["num_classes"], pooling=ck["pooling"])
    model.load_state_dict(ck["state_dict"])
    return model
