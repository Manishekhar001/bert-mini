"""Stages 1 and 2 on top of Hugging Face `Trainer`: masked-language-model training with optional distillation.

DistillTrainer
    student : BertForMaskedLM (stage 2 may wrap query/value with LoRALinear; Trainer checkpoints save that structure)
    teacher : optional frozen BertForMaskedLM (bert-base-uncased) used ONLY in stage 1, on general text
    loss    : w_mlm * MLM cross-entropy
            + w_kd  * KL(teacher || student) on the masked-token distributions (temperature T, scaled by T^2)
            + w_hid * relation distillation between matched layers (see relation_kl)

With teacher=None it is a plain MLM trainer that uses the same memory trick: the 30,522-way output layer is evaluated
only at masked positions (about 15% of tokens), which is what lets stage 1 fit in 6 GB.

Why relation distillation instead of TinyBERT's learned 256->768 projections: in stage 1 the student remains a plain
BertForMaskedLM, so the model can be saved and loaded without custom projection layers. The student's token-to-token
similarity pattern is matched to the teacher's (as in MiniLM), which needs no projection at all. In stage 2 the
query/value LoRA wrappers are used only during training and resume; the final model merges them into standard BERT
linear layers before saving.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM, Trainer, TrainingArguments

from .losses import kd_kl

DEFAULT_LOSS = {"mlm": 1.0, "kd_logits": 2.0, "hidden": 1.0, "temperature": 2.0, "rel_tau": 0.1}


# ---------------------------------------------------------------- loss pieces
def layer_pairs(n_student: int, n_teacher: int) -> list:
    """Student layer i <-> teacher layer i * n_teacher / n_student. 4 student layers vs 12 -> (1,3) (2,6) (3,9) (4,12)."""
    if n_student < 1 or n_teacher < 1:
        raise ValueError("student and teacher must each have at least one encoder layer")
    return [(i, i * n_teacher // n_student) for i in range(1, n_student + 1)]


def relation_kl(s_hidden, t_hidden, mask, tau: float = 0.1):
    """KL between the row-wise softmax of cosine-similarity matrices (token x token) of student and teacher.

    s_hidden [B, L, d_s], t_hidden [B, L, d_t] (the two widths may differ), mask [B, L] with 1 for real tokens.
    """
    if tau <= 0:
        raise ValueError(f"relation temperature must be positive, got {tau}")
    s = F.normalize(s_hidden.float(), dim=-1)
    t = F.normalize(t_hidden.float(), dim=-1)
    s_rel = (s @ s.transpose(1, 2)) / tau
    t_rel = (t @ t.transpose(1, 2)) / tau
    keys = mask[:, None, :].bool()
    s_rel = s_rel.masked_fill(~keys, -1e4)
    t_rel = t_rel.masked_fill(~keys, -1e4)
    kl = F.kl_div(F.log_softmax(s_rel, -1), F.softmax(t_rel, -1), reduction="none").sum(-1)      # [B, L]
    m = mask.float()
    return (kl * m).sum() / m.sum().clamp(min=1.0)


def distill_terms(student, teacher, input_ids, attention_mask, labels, loss_cfg: dict):
    """One batch. Returns (total_loss, {name: detached tensor}). `labels` is -100 except at masked positions."""
    loss_cfg = loss_cfg or {}
    unknown = set(loss_cfg) - set(DEFAULT_LOSS)
    if unknown:
        raise ValueError(f"unknown loss setting(s): {', '.join(sorted(unknown))}")
    w = {**DEFAULT_LOSS, **loss_cfg}
    T = float(w["temperature"])
    if T <= 0:
        raise ValueError(f"logit distillation temperature must be positive, got {T}")
    sel = labels != -100
    use_teacher = teacher is not None
    need_hidden = use_teacher and float(w["hidden"]) > 0

    s_out = student.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=need_hidden)
    if not bool(sel.any()):                                   # no masked token in this batch: contribute nothing
        zero = s_out.last_hidden_state.sum() * 0.0
        return zero, {"mlm": zero.detach()}
    s_logits = student.cls(s_out.last_hidden_state[sel])      # [M, vocab], masked positions only
    target = labels[sel]
    terms = {"mlm": F.cross_entropy(s_logits.float(), target)}
    total = float(w["mlm"]) * terms["mlm"]

    if use_teacher:
        with torch.no_grad():
            t_out = teacher.bert(input_ids=input_ids, attention_mask=attention_mask,
                                 output_hidden_states=need_hidden)
            t_logits = teacher.cls(t_out.last_hidden_state[sel]).float()
        terms["kd"] = kd_kl(s_logits, t_logits, T)
        total = total + float(w["kd_logits"]) * terms["kd"]
        pairs = layer_pairs(student.config.num_hidden_layers, teacher.config.num_hidden_layers)
        if need_hidden:
            hid = sum(relation_kl(s_out.hidden_states[si], t_out.hidden_states[ti], attention_mask, float(w["rel_tau"]))
                      for si, ti in pairs) / len(pairs)
            terms["hid"] = hid
            total = total + float(w["hidden"]) * hid
        terms["t_acc"] = (t_logits.argmax(-1) == target).float().mean()

    terms["acc"] = (s_logits.argmax(-1) == target).float().mean()
    return total, {k: v.detach() for k, v in terms.items()}


# ---------------------------------------------------------------- models and arguments
def load_teacher(name: str, device, dtype=torch.bfloat16) -> BertForMaskedLM:
    t = BertForMaskedLM.from_pretrained(name).to(dtype).to(device).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def make_training_args(cfg: dict, out_dir: str, seed: int = 42) -> TrainingArguments:
    """cfg['args'] is passed to TrainingArguments as written in the YAML (names are the standard HF ones)."""
    a = dict(cfg["args"])
    a.setdefault("eval_strategy", "steps")
    a.setdefault("save_strategy", "steps")
    a.setdefault("report_to", "none")
    if a.get("bf16") and not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8):
        print("[warn] bf16 requested but this device does not support it; training in fp32")
        a["bf16"] = False
    return TrainingArguments(output_dir=out_dir, seed=seed, remove_unused_columns=False, **a)


# ---------------------------------------------------------------- trainer
class DistillTrainer(Trainer):
    """HF Trainer with a custom loss. Pass `teacher=` for distillation, omit it for plain MLM."""

    def __init__(self, *args, teacher=None, loss_cfg: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.model_accepts_loss_kwargs = False   # our loss is already a per-batch mean; Trainer divides by accum
        except AttributeError:                       # property in some future version
            pass
        self.loss_cfg = {**DEFAULT_LOSS, **(loss_cfg or {})}
        self.teacher = teacher
        if teacher is not None:
            self.teacher = teacher.to(self.args.device).eval()
        self._acc, self._acc_n = {}, 0

    # -- loss
    def _loss(self, model, inputs):
        base = model.module if hasattr(model, "module") else model
        # We call base.bert / base.cls directly (not model.forward), which skips the autocast that Trainer/accelerate
        # attach to forward(), so apply bf16 autocast explicitly. Single-GPU only (no DDP).
        dev = self.args.device.type
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=bool(self.args.bf16) and dev == "cuda"):
            return distill_terms(base, self.teacher, inputs["input_ids"], inputs["attention_mask"], inputs["labels"],
                                 self.loss_cfg)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        total, terms = self._loss(model, inputs)
        if model.training:
            for k, v in terms.items():
                self._acc[k] = self._acc.get(k, 0.0) + v
            self._acc_n += 1
        return (total, {"loss": total.detach()}) if return_outputs else total

    # -- logging: add the loss components (mlm / kd / hid / acc ...) averaged since the previous log line
    def log(self, logs, *args, **kwargs):
        if "loss" in logs and self._acc_n:
            logs = {**logs, **{k: float(v) / self._acc_n for k, v in self._acc.items()}}
            self._acc, self._acc_n = {}, 0
        return super().log(logs, *args, **kwargs)

    # -- evaluation: masked-token metrics, one entry per eval set; masks are fixed so numbers are comparable
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        sets = eval_dataset if eval_dataset is not None else self.eval_dataset
        sets = sets if isinstance(sets, dict) else {None: sets}
        metrics, t0 = {}, time.time()
        for name, ds in sets.items():
            prefix = metric_key_prefix if name is None else f"{metric_key_prefix}_{name}"
            metrics.update({f"{prefix}_{k}": v for k, v in self._eval_one(ds).items()})
        metrics[f"{metric_key_prefix}_runtime"] = time.time() - t0
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def _eval_one(self, ds) -> dict:
        model = self.model
        was_training = model.training
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        sums, n = {}, 0
        try:
            model.eval()
            # The MLM collator masks randomly; reset RNG so every validation pass scores the same tokens.
            torch.manual_seed(1234)
            loader = self.get_eval_dataloader(ds)
            with torch.no_grad():
                for batch in loader:
                    batch = self._prepare_inputs(batch)
                    with self.compute_loss_context_manager():
                        _, terms = self._loss(model, batch)
                    bs = batch["input_ids"].size(0)
                    for k, v in terms.items():
                        sums[k] = sums.get(k, 0.0) + float(v) * bs
                    n += bs
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            model.train(was_training)
        return {k: v / max(n, 1) for k, v in sums.items()}


# ---------------------------------------------------------------- speed / memory check
def benchmark_distill(student_cfg: dict, teacher_name: str | None, loss_cfg: dict, micro_batch: int, block_len: int,
                      steps: int = 15, log=print) -> dict:
    """A few real optimisation steps on random tokens: sequences per second and peak GPU memory."""
    from .model import build_student_mlm
    from .utils import get_device

    if micro_batch < 1 or block_len < 2 or steps <= 3:
        raise ValueError("benchmark needs micro_batch >= 1, block_len >= 2, and steps > 3")
    device = get_device()
    student = build_student_mlm(student_cfg).to(device).train()
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    teacher = (load_teacher(teacher_name, device, torch.bfloat16 if use_bf16 else torch.float32)
               if teacher_name else None)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-4)
    vocab_size = int(student.config.vocab_size)
    ids = torch.randint(0, vocab_size, (micro_batch, block_len), device=device)
    ids[:, 0], ids[:, -1] = min(101, vocab_size - 1), min(102, vocab_size - 1)
    attention_mask = torch.ones_like(ids)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = None
    for i in range(steps):
        if i == 3:                                            # skip warm-up steps
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        labels = torch.full_like(ids, -100)
        pick = torch.rand(ids.shape, device=device) < 0.15
        pick[:, 0] = pick[:, -1] = False
        labels[pick] = ids[pick]
        inp = ids.clone()
        inp[pick] = min(103, vocab_size - 1)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            loss, _ = distill_terms(student, teacher, inp, attention_mask, labels, loss_cfg)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    res = {"seq_per_s": (steps - 3) * micro_batch / dt, "micro_batch": micro_batch,
           "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None}
    res["hours_for_1M_blocks"] = 1e6 / res["seq_per_s"] / 3600
    log(res)
    return res
