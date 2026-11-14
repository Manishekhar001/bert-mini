"""Stage 2: domain adaptation with MLM and query/value LoRA on ASRS narratives. No teacher.

    python stage2/train.py                                   # uses runs/stage1_distill/final
    python stage2/train.py --init runs/stage1_mlm/final --out runs/stage2_domain_mlm   # control run
    python stage2/train.py --max_len 256                      # keep the original 256 context instead of extending
    python stage2/train.py --resume                           # resume adapters from the newest Trainer checkpoint

Context: stage 1 used 256 positions (the teacher setup). Here the position table is extended to --max_len (default 512)
by copying the learned rows, so long reports are not cut. Each report is one training example (no cross-report packing),
exactly like the fine-tuning stage will see it.

Text used: TRAIN-split narratives only (validation narratives for monitoring). Test reports are never used.
The Trainer checkpoints retain the LoRA wrappers for resume. The final adapters are merged into the base query/value
weights, so <out>/final is a normal BertForMaskedLM save_pretrained folder that Stage 3 can load.
"""


import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.text_utils import tokenize_texts
from common.utils import load_config, read_jsonl, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="stage2/config.yaml")
    ap.add_argument("--init", default=None, help="overrides init (folder or Hub id of a BertForMaskedLM)")
    ap.add_argument("--out", default=None, help="overrides out_dir")
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--data_dir", default="data/asrs")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()


    from datasets import Dataset, load_from_disk
    from transformers import BertTokenizerFast, DataCollatorForLanguageModeling
    from transformers.trainer_utils import get_last_checkpoint

    from common.distill import DistillTrainer, make_training_args
    from common.model import apply_lora_to_mlm ,  extend_positions, load_student_mlm, merge_lora_mlm


    cfg = load_config(args.config , args.set)
    init =  args.init or cfg['init_from']
    out = Path(args.out or cfg['out_dir'])

    out.mkdir(parents=True , exist_ok=True)
    max_len = args.max_len if args.max_len is not None else cfg['max_len']

    if max_len< 2:
        raise SystemError("--max_len must be at least 2 to fit [cls] and [SEP]")

    head_tokens = cfg['head_tokens'] if max_len == cfg['max_len'] else int(max_len * 0.75)

    tok = BertTokenizerFast.from_pretrained(cfg['tokenizer'])
    student = load_student_mlm(init)
    old = student.config.max_position_embeddings
    extend_positions(student, max_len)

    print(f"init {init}: positions {old} -> {student.config.max_position_embeddings}")
    use_lora = bool(cfg.get("use_lora",True))

    if use_lora:
        student = apply_lora_to_mlm(
            student,
            r = int(cfg.get("lora_r",8)),
            alpha = float(cfg.get("lora_alpha" , 16.0)),
            dropout = float(cfg.get("lora_dropout" , 0.05)),
            target_modules=tuple(cfg.get("lora_target_modules", ["query", "value"]))
        )
    else:
        for parameter in student.parameters():
            parameter.requires_grad_(True)
        print("[Lora] disabled by config; all MLM parameters will be trainable")

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())

    print(f'Stage 2 Trainable parameters : {trainable:,}/{total:,} ({trainable / total:.2%})')

    d = Path(args.data_dir)
    train_rows , val_rows = (read_jsonl(d / f"{s}.jsonl") for s in ('train' , 'validation'))
    train_texts = [r['text'] for r in train_rows]
    val_texts = [r['text'] for r in val_rows]

    def to_ds(texts):
        return Dataset.from_dict({'input_ids' : tokenize_texts(tok, texts, max_len, head_tokens)})

    train , val = to_ds(train_texts) , to_ds(val_texts)
    evals = {'asrs_val'  :val}
    gen_dir = cfg['data'].get('general_val_dir')
    if gen_dir and Path(gen_dir).exists():
        # for getting check on general text (256 - token blocks)

        g = load_from_disk(gen_dir)
        evals['general'] = g.select(range(min(200 , len(g))))
    print(f"train reports: {len(train):,} | validation reports: {len(val):,} | max_len {max_len}")

    # Avoid padding beyond max_len for custom windows that are not divisible by eight.

    collator = DataCollatorForLanguageModeling( tok, mlm = True, mlm_probability=float(cfg['mlm_prob']))
    trainer = DistillTrainer(model = student, args = make_training_args(cfg, str(out), cfg['seed']) , train_dataset = train , eval_dataset = evals , data_collator = collator , teacher = None, loss_cfg={'mlm' : 1.0})

    ckpt = get_last_checkpoint(str(out)) if args.resume else None
    if args.resume and ckpt is None:
        raise FileNotFoundError(f"--resume was requested, but no Trainer checkpoint exists in {out}")
    trainer.train(resume_from_checkpoint=ckpt)


    final = out / 'final'
    if use_lora :
        # checkpoints stay adapter-wrapped so Trainer can resume ;the delivered artifact is standard bert

        merge_lora_mlm(trainer.model)
    trainer.save_model(str(final))                                # best checkpoint (lowest ASRS validation MLM loss)
    tok.save_pretrained(str(final))
    write_json(out / "log_history.json", trainer.state.log_history)
    print(f"done. domain-adapted student saved to {final}")


if __name__ == "__main__":
    main()