"""Small helpers shared by all scripts. Importing this module never imports torch."""
from __future__ import annotations

import contextlib
import json
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------- config
def load_yaml(path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _coerce(raw: str):
    import yaml

    val = yaml.safe_load(raw)
    if isinstance(val, str):  # PyYAML reads 3e-4 as a string; make it a float
        try:
            return float(val)
        except ValueError:
            return val
    return val


def apply_overrides(cfg: dict, overrides) -> dict:
    """Apply 'a.b.c=value' overrides (value parsed as YAML, e.g. train.max_steps=200)."""
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, got {item!r}")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(raw)
    return cfg


def load_config(path, overrides=None) -> dict:
    return apply_overrides(load_yaml(path), overrides)


# ---------------------------------------------------------------- json
class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_Enc)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, cls=_Enc, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- torch helpers (lazy import)
def get_device(prefer: str = "auto"):
    import torch

    if prefer == "auto":
        prefer = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(prefer)


def autocast_ctx(device):
    """bf16 autocast on Ampere+ GPUs (RTX 30xx), plain fp32 otherwise (CPU, old GPUs)."""
    import torch

    if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:   # Ampere or newer (RTX 30xx)
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def stage3_uses_lora(init: str, configured: bool) -> bool:
    """Never freeze a random encoder behind adapters in the Stage 3 random-init ablation."""
    return bool(configured) and init != "random"
