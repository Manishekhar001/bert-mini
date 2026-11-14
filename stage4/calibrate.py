"""Fit probability calibration, a review cutoff and an optional out-of-domain warning.

Calibration is fitted on validation reports. Test reports are only used for the final report.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.dataset import AsrsDataset
from common.metrics import auroc, evaluate, logits_to_probs, reliability_bins
from common.schema import Schema
from common.utils import get_device, read_json, read_jsonl, write_json
from stage4.calibration import calibrated_probs, domain_score, fit_temperature, threshold_for_accuracy
from stage4.inference import torch_logits_for_texts


def load_ood_texts(args):
    if args.ood_file:
        with open(args.ood_file, encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()][:args.ood_n]
    from datasets import load_dataset

    name, _, config = args.ood_hf.partition(":")
    dataset = load_dataset(name, config or None, split="train", streaming=True).shuffle(seed=0, buffer_size=5000)
    texts = []
    for row in dataset:
        text = str(row.get("text", "")).strip()
        if len(text) > 200:
            texts.append(text[:1500])
        if len(texts) >= args.ood_n:
            break
    return texts


def reliability_bins_for_report(confidence, correct, n_bins=10):
    return [{"confidence": c, "accuracy": a, "count": n}
            for c, a, n in reliability_bins(confidence, correct, n_bins)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Stage 3 run folder")
    parser.add_argument("--data_dir", default="data/asrs")
    parser.add_argument("--out", default="artifacts/final")
    parser.add_argument("--target_acc", type=float, default=0.90,
                        help="target accuracy among validation predictions accepted automatically")
    parser.add_argument("--ood_hf", default="wikimedia/wikipedia:20231101.en")
    parser.add_argument("--ood_file", default=None, help="optional newline-separated outside-domain texts")
    parser.add_argument("--ood_n", type=int, default=2000)
    parser.add_argument("--no_ood", action="store_true")
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()

    from transformers import BertTokenizerFast
    from common.model import load_classifier

    run, data_dir, out = Path(args.run), Path(args.data_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    schema = Schema.load(run / "schema_resolved.json")
    training_meta = read_json(run / "metrics.json")
    data_cfg = training_meta["data_cfg"]
    tokenizer_name = args.tokenizer or data_cfg.get("tokenizer", "bert-base-uncased")
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    val = AsrsDataset(read_jsonl(data_dir / "validation.jsonl"), schema, tokenizer,
                      data_cfg["max_len"], data_cfg["head_tokens"])
    test = AsrsDataset(read_jsonl(data_dir / "test.jsonl"), schema, tokenizer,
                       data_cfg["max_len"], data_cfg["head_tokens"])
    val_logits = np.load(run / "logits_val.npz")["logits"]
    test_logits = np.load(run / "logits_test.npz")["logits"]

    # 1) Temperature scaling adjusts the sharpness of probabilities without changing the winning class.
    temperature = fit_temperature(val_logits, val.labels)
    val_probs = calibrated_probs(val_logits, temperature)
    test_probs_raw = logits_to_probs(test_logits)
    test_probs = calibrated_probs(test_logits, temperature)
    before = evaluate(test_probs_raw, test.labels)
    after = evaluate(test_probs, test.labels)

    # 2) Pick confidence cutoff using validation only. Low-confidence reports are sent to a human.
    val_correct = val_probs.argmax(axis=1) == val.labels
    val_confidence = val_probs.max(axis=1)
    cutoff, val_coverage, val_accuracy = threshold_for_accuracy(val_confidence, val_correct, args.target_acc)
    test_correct = test_probs.argmax(axis=1) == test.labels
    test_confidence = test_probs.max(axis=1)
    accepted = test_confidence >= cutoff if cutoff is not None else np.zeros(len(test), dtype=bool)
    test_coverage = float(accepted.mean()) if len(test) else 0.0
    test_accepted_accuracy = float(test_correct[accepted].mean()) if accepted.any() else None

    calibration = {
        "temperature": temperature,
        "target_accuracy": args.target_acc,
        "confidence_threshold": cutoff,
        "data_cfg": data_cfg,
    }
    report = {
        "run": str(run),
        "temperature": temperature,
        "before_calibration": before,
        "after_calibration": after,
        "review": {
            "target_accuracy": args.target_acc,
            "confidence_threshold": cutoff,
            "validation_coverage": val_coverage,
            "validation_accuracy_at_threshold": val_accuracy,
            "test_coverage": test_coverage,
            "test_accuracy_when_accepted": test_accepted_accuracy,
            "test_reliability_bins": reliability_bins_for_report(test_confidence, test_correct),
        },
    }

    # 3) Optional OOD warning. Compare two simple scores, then keep the one with better validation AUROC.
    if not args.no_ood:
        try:
            device = get_device()
            model = load_classifier(run / "best.pt").to(device)
            texts = load_ood_texts(args)
            if not texts:
                raise ValueError("the outside-domain text source returned no usable texts")
            ood_logits = torch_logits_for_texts(model, tokenizer, texts, device,
                                                data_cfg["max_len"], data_cfg["head_tokens"])
            ood_probs = calibrated_probs(ood_logits, temperature)
            candidates = []
            for kind in ("confidence", "energy"):
                inside_scores = domain_score(kind, val_logits, val_probs)
                outside_scores = domain_score(kind, ood_logits, ood_probs)
                score_auc = auroc(inside_scores, outside_scores)
                boundary = float(np.percentile(inside_scores, 5))
                candidates.append({
                    "score": kind, "threshold": boundary, "auroc": score_auc,
                    "in_domain_acceptance": float(np.mean(inside_scores >= boundary)),
                    "outside_rejected": float(np.mean(outside_scores < boundary)),
                    "n_outside": len(outside_scores),
                })
            selected = max(candidates, key=lambda item: item["auroc"])
            calibration["ood"] = selected
            report["ood"] = {"candidates": candidates, "selected": selected}
            print(f"OOD uses {selected['score']} (AUROC {selected['auroc']:.3f}); "
                  f"validation boundary {selected['threshold']:.4g}; "
                  f"rejects {selected['outside_rejected']:.1%} of this outside-text sample")
        except Exception as error:
            print(f"[warning] OOD calibration skipped: {error}")

    shutil.copy(run / "best.pt", out / "model.pt")
    shutil.copy(run / "schema_resolved.json", out / "schema_resolved.json")
    write_json(out / "calibration.json", calibration)
    write_json(out / "report.json", report)
    tokenizer.save_pretrained(str(out / "tokenizer"))
    print(f"temperature {temperature:.4f}; confidence cutoff {cutoff}; "
          f"test accuracy {after['accuracy']:.3f}, macro-F1 {after['macro_f1']:.3f}; package -> {out}")


if __name__ == "__main__":
    main()
