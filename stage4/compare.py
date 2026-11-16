"""Collect classifier results into one Markdown file (results.md).

Sections: raw test results, the same models after a per-class prior correction fitted on validation (applied to
every model, so the comparison is fair), per-class results and a confusion matrix for the packaged model, and CPU
latency. Test labels are only used for scoring; every fitted number comes from validation.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.metrics import evaluate, macro_f1
from common.schema import Schema
from common.utils import read_json, read_jsonl
from stage4.analysis import confusion_matrix, markdown_confusion, markdown_per_class, per_class_rows
from stage4.calibration import (apply_bias, calibrated_probs, choose_bias_strength, fit_class_bias,
                                fit_temperature)


def load_labels(path) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in read_jsonl(path)], dtype=np.int64)


def load_logits(run: Path):
    """(validation logits, test logits) of a run, or None when the run did not save them."""
    val, test = run / "logits_val.npz", run / "logits_test.npz"
    if val.exists() and test.exists():
        return np.load(val)["logits"], np.load(test)["logits"]
    return None


def log_softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def corrected(val_logits, val_labels, test_logits, test_labels, l2: float = 1e-2):
    """Prior correction + temperature, both fitted on validation. Returns (strength, test metrics, test logits)."""
    strength, _ = choose_bias_strength(val_logits, val_labels, l2=l2)
    bias = strength * fit_class_bias(val_logits, val_labels, l2) if strength > 0 else None
    val_adjusted, test_adjusted = apply_bias(val_logits, bias), apply_bias(test_logits, bias)
    temperature = fit_temperature(val_adjusted, val_labels)
    metrics = evaluate(calibrated_probs(test_adjusted, temperature), test_labels)
    return strength, metrics, test_adjusted


def blend_weight(first_val, second_val, val_labels, n_classes: int):
    """Weight on the second model's log-probabilities, chosen on validation by 0.5 * accuracy + 0.5 * macro-F1."""
    best_weight, best_score = 0.0, -1.0
    for weight in np.linspace(0.0, 1.0, 11):
        mixed = (1 - weight) * log_softmax(first_val) + weight * log_softmax(second_val)
        predicted = mixed.argmax(axis=1)
        score = 0.5 * float((predicted == val_labels).mean()) + 0.5 * macro_f1(val_labels, predicted, n_classes)
        if score > best_score + 1e-9:
            best_weight, best_score = float(weight), score
    return best_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", default="artifacts/final")
    parser.add_argument("--data_dir", default="data/asrs")
    parser.add_argument("--out", default="results.md")
    args = parser.parse_args()

    data_dir, final = Path(args.data_dir), Path(args.final)
    schema = Schema.load(data_dir / "schema_resolved.json")
    val_labels, test_labels = load_labels(data_dir / "validation.jsonl"), load_labels(data_dir / "test.jsonl")
    majority = float(np.bincount(test_labels, minlength=schema.n_classes).max() / len(test_labels))

    runs = []
    for path in sorted(Path("runs").glob("*/metrics.json")):
        metrics = read_json(path)
        if "test" in metrics:
            metrics["_dir"] = path.parent
            runs.append(metrics)
    runs.sort(key=lambda item: item["test"].get("macro_f1", float("-inf")), reverse=True)

    lines = ["## Test-set results (raw model output)", "",
             f"Always guessing the most common class scores {majority:.3f} accuracy on this test set. "
             "Validation and test keep the natural class mix; training is rebalanced, so the raw numbers below "
             "are not the best these models can do (see the next table).", "",
             "| model | parameters (M) | accuracy | macro-F1 | ECE |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for item in runs:
        test = item["test"]
        params = item.get("params")
        params = "" if params is None else f"{params / 1e6:.1f}"
        lines.append(f"| {item['name']} | {params} | {test['accuracy']:.3f} | "
                     f"{test['macro_f1']:.3f} | {test['ece']:.3f} |")

    # Same prior correction for every model that saved its logits, so no model gets an unfair advantage.
    fixed_rows, saved = [], {}
    for item in runs:
        logits = load_logits(item["_dir"])
        if logits is None:
            continue
        saved[item["name"]] = logits
        strength, metrics, _ = corrected(logits[0], val_labels, logits[1], test_labels)
        fixed_rows.append((item["name"], f"{strength:.2f}", metrics))
    tfidf = next((name for name in saved if name.startswith("tfidf")), None)
    best_neural = next((item["name"] for item in runs
                        if item["name"] in saved and not item["name"].startswith("tfidf")), None)
    if tfidf and best_neural:
        weight = blend_weight(saved[best_neural][0], saved[tfidf][0], val_labels, schema.n_classes)
        mixed_val = (1 - weight) * log_softmax(saved[best_neural][0]) + weight * log_softmax(saved[tfidf][0])
        mixed_test = (1 - weight) * log_softmax(saved[best_neural][1]) + weight * log_softmax(saved[tfidf][1])
        strength, metrics, _ = corrected(mixed_val, val_labels, mixed_test, test_labels)
        fixed_rows.append((f"{best_neural} + {tfidf} blend (weight {weight:.1f} on {tfidf}; analysis only)",
                           f"{strength:.2f}", metrics))
    if fixed_rows:
        lines += ["", "## After per-class prior correction and temperature scaling (fitted on validation)", "",
                  "| model | correction strength | accuracy | macro-F1 | ECE |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for name, strength, metrics in sorted(fixed_rows, key=lambda row: -row[2]["macro_f1"]):
            lines.append(f"| {name} | {strength} | {metrics['accuracy']:.3f} | "
                         f"{metrics['macro_f1']:.3f} | {metrics['ece']:.3f} |")
        lines += ["", "Strength 1.00 applies the whole fitted correction and 0.00 none; it is chosen by "
                      "cross-validation inside the validation set. The blend row mixes two models' "
                      "log-probabilities and is not part of the packaged model."]

    # Per-class table and confusion matrix for the packaged model (including its stored class_bias).
    report_path, calibration_path = final / "report.json", final / "calibration.json"
    if report_path.exists() and calibration_path.exists():
        run = Path(read_json(report_path)["run"])
        logits = load_logits(run)
        if logits is not None:
            bias = read_json(calibration_path).get("class_bias")
            predicted = apply_bias(logits[1], bias).argmax(axis=1)
            matrix = confusion_matrix(test_labels, predicted, schema.n_classes)
            rows = per_class_rows(matrix, schema.classes)
            lines += ["", f"## Per-class test results: packaged model ({run.name})", "",
                      markdown_per_class(rows),
                      "", "## Confusion matrix: packaged model", "",
                      "Each row is a true class; each cell is the percent of that class predicted as the "
                      "column's class (bold = correct).", "",
                      markdown_confusion(matrix, schema.classes)]

    benchmark = final / "bench.json"
    if benchmark.exists():
        bench = read_json(benchmark)
        lines += ["", f"## CPU latency ({bench.get('threads')} threads)", "",
                  "| backend | size (MB) | batch-1 median (ms) | batch-1 p95 (ms) | batch-32 (ms/report) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for name, result in bench.items():
            if isinstance(result, dict) and "batch1" in result:
                size = f"{result['size_mb']:.1f}" if "size_mb" in result else ""
                lines.append(f"| {name} | {size} | {result['batch1']['median_ms_per_call']:.1f} | "
                             f"{result['batch1']['p95_ms_per_call']:.1f} | "
                             f"{result['batch32']['ms_per_report']:.2f} |")
    result_text = "\n".join(lines) + "\n"
    Path(args.out).write_text(result_text, encoding="utf-8")
    print(result_text)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
