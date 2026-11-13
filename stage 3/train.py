"""Stage 3: train one shared report classifier with one mutually-exclusive class target.

Example: python stage3/train.py --run C_distill_domain --init runs/stage2_domain/final
"""
# ---------------------------------------------------------------
# STANDARD LIBRARY IMPORTS
# ---------------------------------------------------------------
import argparse          # reads options typed in the terminal (--run, --init, ...)
import shutil            # copy files and delete folders
import sys               # lets us change Python's search path for imports
from pathlib import Path # easy, safe way to work with file/folder paths

# Add the project's root folder to Python's search path.
# __file__ = this script; .parents[1] = the folder above "stage3".
# Without this, "from common..." would fail when running "python stage3/train.py".
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # numbers/arrays library

# Small helper functions that live in the project's own "common" folder
from common.utils import count_params, load_config, read_jsonl, set_seed, stage3_uses_lora, write_json


def main():
    # -----------------------------------------------------------
    # 1) READ COMMAND-LINE OPTIONS
    # -----------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)   # name of this experiment (output folder name)
    parser.add_argument("--init", required=True,
                        help="random | local BERT checkpoint | hf:<BERT model id>")  # where the weights start from
    parser.add_argument("--config", default="stage3/config.yaml")  # settings file
    parser.add_argument("--data_dir", default="data/asrs")         # folder with the data
    parser.add_argument("--seed", type=int, default=None)          # optional random seed override
    parser.add_argument("--set", action="append", default=[])      # override config values, can repeat
    args = parser.parse_args()  # actually read what the user typed

    # -----------------------------------------------------------
    # 2) HEAVY IMPORTS (done here so --help stays fast)
    # -----------------------------------------------------------
    import torch
    from transformers import BertTokenizerFast, EarlyStoppingCallback

    from common.dataset import AsrsDataset
    from common.distill import make_training_args
    from common.metrics import evaluate, logits_to_probs
    from common.model import ReportClassifier, build_encoder, save_classifier
    from common.schema import Schema
    from stage3.sft import ClassifierTrainer, SftCollator, SftDataset, class_weights, make_compute_metrics, predict_logits

    # -----------------------------------------------------------
    # 3) LOAD SETTINGS, SET SEED, PREPARE FOLDERS
    # -----------------------------------------------------------
    cfg = load_config(args.config, args.set)                       # read YAML + apply --set overrides
    seed = args.seed if args.seed is not None else cfg["seed"]     # command-line seed wins, else config seed
    set_seed(seed)                                                 # fix randomness so results can be repeated
    data_dir, out = Path(args.data_dir), Path("runs") / args.run   # input folder, output folder
    out.mkdir(parents=True, exist_ok=True)                         # create output folder if missing

    # -----------------------------------------------------------
    # 4) LOAD CLASS INFO, TOKENIZER AND DATA
    # -----------------------------------------------------------
    schema = Schema.load(data_dir / "schema_resolved.json")        # list of classes + how many examples each has
    tokenizer = BertTokenizerFast.from_pretrained(cfg["tokenizer"])  # turns text into token ids
    # Read train / validation / test files (one JSON object per line)
    rows = {split: read_jsonl(data_dir / f"{split}.jsonl") for split in ("train", "validation", "test")}
    # Convert raw rows into tokenized datasets (cut to max_len; head_tokens = how many tokens kept from the start)
    data = {split: AsrsDataset(rows[split], schema, tokenizer, cfg["max_len"], cfg["head_tokens"])
            for split in rows}
    # What fraction of training reports are as long as (so were cut by) the max length?
    truncated = np.mean([len(ids) >= cfg["max_len"] for ids in data["train"].input_ids])
    print(f"{schema.n_classes} classes, {len(data['train']):,} training reports; "
          f"{cfg['max_len']}-token window truncates {truncated:.1%} of reports")

    # -----------------------------------------------------------
    # 5) BUILD THE MODEL
    # -----------------------------------------------------------
    # The encoder is the BERT body; --init decides its starting weights
    encoder = build_encoder(args.init, cfg["student"], max_positions=cfg["max_len"])
    # Put a classification head on top (pooling = how to squash tokens into one vector)
    model = ReportClassifier(encoder, schema.n_classes, cfg["pooling"], cfg["dropout"])
    deployed_params = count_params(model)  # parameter count of the final model (before any LoRA extras)
    # Decide whether to use LoRA (small add-on layers instead of training the whole encoder)
    lora_enabled = stage3_uses_lora(args.init, cfg.get("use_lora", False))
    if cfg.get("use_lora", False) and args.init == "random":
        # LoRA only makes sense when starting from pretrained weights
        print("[lora] disabled for random initialization; the full encoder will train")
    if lora_enabled:
        model.apply_lora(
            r=int(cfg.get("lora_r", 8)),                 # LoRA size (rank)
            alpha=float(cfg.get("lora_alpha", 16.0)),    # LoRA strength scaling
            dropout=float(cfg.get("lora_dropout", 0.05)),  # dropout inside LoRA
            target_modules=tuple(cfg.get("lora_target_modules", ["query", "value"])),  # which layers get LoRA
        )
    all_params = sum(p.numel() for p in model.parameters())                          # total parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # ones that will learn
    strategy = "query/value LoRA" if lora_enabled else "full encoder fine-tuning"
    print(f"init={args.init}; {strategy}; {trainable_params:,}/{all_params:,} trainable parameters")

    # -----------------------------------------------------------
    # 6) SET UP THE TRAINER AND TRAIN
    # -----------------------------------------------------------
    weights = class_weights(schema.counts, cfg.get("class_weight"))  # optional: give rare classes more importance
    trainer = ClassifierTrainer(
        model=model,
        args=make_training_args(cfg, str(out / "trainer"), seed),    # epochs, batch size, learning rate, etc.
        train_dataset=SftDataset(data["train"]),                     # data used to learn
        eval_dataset=SftDataset(data["validation"]),                 # data used to check progress
        data_collator=SftCollator(tokenizer.pad_token_id or 0),      # builds equal-length batches (pads short ones)
        compute_metrics=make_compute_metrics(),                      # accuracy / macro-F1 / ECE during evaluation
        # Stop early if validation stops improving for N checks in a row (default 4)
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(cfg.get("early_stopping_patience", 4)))],
        class_weights=weights,                                       # passed to our custom loss
        head_lr=float(cfg.get("head_lr", cfg["args"]["learning_rate"])),  # separate learning rate for the head
    )
    trainer.train()  # <-- the actual training happens here

    # -----------------------------------------------------------
    # 7) PREDICT WITH THE FINAL MODEL
    # -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if there is one
    model = trainer.model.to(device)
    batch_size = int(cfg["args"].get("per_device_eval_batch_size", 64))
    val_logits = predict_logits(model, data["validation"], device, batch_size)  # raw scores on validation set
    test_logits = predict_logits(model, data["test"], device, batch_size)       # raw scores on test set

    # -----------------------------------------------------------
    # 8) SAVE EVERYTHING
    # -----------------------------------------------------------
    save_classifier(out / "best.pt", model.cpu())                       # save the model weights
    np.savez_compressed(out / "logits_val.npz", logits=val_logits)      # save validation scores
    np.savez_compressed(out / "logits_test.npz", logits=test_logits)    # save test scores
    shutil.copy(data_dir / "schema_resolved.json", out / "schema_resolved.json")  # keep class list with the model
    result = {
        "name": args.run, "init": args.init, "seed": seed, "params": deployed_params,
        "size_mb_fp32": deployed_params * 4 / 1e6,   # each fp32 number = 4 bytes -> size in megabytes
        "training": {"strategy": strategy, "lora_enabled": lora_enabled,
                     "trainable_params": trainable_params, "total_params": all_params},
        "data_cfg": {"max_len": cfg["max_len"], "head_tokens": cfg["head_tokens"],
                     "tokenizer": cfg["tokenizer"]},
        "history": trainer.state.log_history,         # loss/metrics logged during training
        "val": evaluate(logits_to_probs(val_logits), data["validation"].labels),   # final validation scores
        "test": evaluate(logits_to_probs(test_logits), data["test"].labels),       # final test scores
    }
    write_json(out / "metrics.json", result)                 # save the summary as a JSON file
    shutil.rmtree(out / "trainer", ignore_errors=True)       # delete temporary checkpoint folder
    print(f"\nTest accuracy {result['test']['accuracy']:.4f}; macro-F1 {result['test']['macro_f1']:.4f}; "
          f"ECE {result['test']['ece']:.4f}")
    print(f"saved to {out}")


# Run main() only when this file is executed directly (not when imported)
if __name__ == "__main__":
    main()