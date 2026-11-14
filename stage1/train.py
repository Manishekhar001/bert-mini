"""Stage 1: general-knowledge distillation. A frozen bert-base-uncased teaches a random-init BERT-mini on general text.

    python stage1/train.py --overfit                       # ~1 min sanity check: loss must fall fast
    python stage1/train.py                                 # the real run (hours; resumable with --resume)
    # plain-MLM control for the ablation (no teacher):
    python stage1/train.py --set objective=mlm --set out_dir=runs/stage1_mlm

Built on Hugging Face Trainer (checkpointing, resume, logging, mixed precision). Output: <out_dir>/final (a normal
save_pretrained folder, loadable with BertForMaskedLM.from_pretrained) + the tokenizer.
"""

# Continue for about one more pass from the completed Stage 1 model.
# This starts a new optimizer schedule and writes to a separate run folder.
# uv run python stage1/train.py --set init_from=runs/stage1_distill/final --set out_dir=runs/stage1_distill_epoch2 --set seed=43 --set args.max_steps=7800 --set args.learning_rate=1.0e-4
# Then start Stage 2 from the continued Stage 1 model:
# uv run python stage2/train.py --init runs/stage1_distill_epoch2/final


import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import count_params, load_config, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="stage1/config.yaml")
    ap.add_argument("--resume", action="store_true", help="continue from the newest checkpoint in out_dir")
    ap.add_argument("--overfit", action="store_true", help="train on a handful of blocks only (sanity check)")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()


    import torch
    from datasets import load_from_disk
    from transformers import BertTokenizerFast, DataCollatorForLanguageModeling 
    from transformers.trainer_utils import get_last_checkpoint
    # data collater for language modeling -> it prepares the dataset for the mlm task.

    from common.distill import DistillTrainer, load_teacher, make_training_args
    from common.model import load_student_mlm, build_student_mlm

    cfg = load_config(args.config, args.set)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    distill = cfg['objective'] == 'distill'

    tok = BertTokenizerFast.from_pretrained(cfg['tokenizer'])
    student = (load_student_mlm(cfg["init_from"])) if cfg.get("init_from") else build_student_mlm(cfg['student'])

    # load_student_mlm doesn't start from scratch

    teacher = None
    if distill:
        use_bf16 = (bool(cfg['args'].get("bf16")) and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)

        teacher = load_teacher(cfg['teacher'], "cpu" , torch.bfloat16 if use_bf16 else torch.float32)

        if teacher.config.vocab_size != student.config.vocab_size:
            raise SystemExit("Student and teacher must share the tokenizer vocabulary")

    print(f"Student : {count_params(student) / 1e6:.2f}M parameters | objective: {cfg['objective']}")

    train = load_from_disk(cfg['data']['train_dir'])
    n_eval = int(cfg['data'].get('eval_blocks',5000))
    evals = {'general' : load_from_disk(cfg['data']['val_dir'])}
    evals = {k: v.select(range(min(n_eval, len(v)))) for k, v in evals.items()}

    if args.overfit:
        train = train.select(range(2*int(cfg['args']['per_device_train_batch_size'])))

        cfg['args'].update(max_steps = 200, logging_steps = 10, eval_steps = 100, save_steps = 200)

    print(f"train blocks: {len(train):,} | eval sets: { {k: len(v) for k, v in evals.items()} }")

    collator = DataCollatorForLanguageModeling(tok, mlm = True, mlm_probability=float(cfg['mlm_prob']))
    trainer = DistillTrainer(model = student, args = make_training_args(cfg, str(out) , cfg['seed']), train_dataset = train, eval_dataset = evals, data_collator = collator, teacher = teacher, loss_cfg = cfg['loss'])

    ckpt = get_last_checkpoint(str(out)) if args.resume else None
    trainer.train(resume_from_checkpoint=ckpt)

    final = out / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    write_json(out / "log_history.json", trainer.state.log_history)
    print(f"done. student saved to {final}")


if __name__ == "__main__":
    main()