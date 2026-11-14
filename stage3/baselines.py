"""Simple TF-IDF + logistic-regression baseline for the one multiclass label."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.metrics import evaluate, logits_to_probs
from common.schema import Schema
from common.utils import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/asrs")
    parser.add_argument("--out", default="runs/tfidf")
    parser.add_argument("--C", type=float, default=4.0, help="inverse regularization strength")
    parser.add_argument("--max_features", type=int, default=200_000)
    args = parser.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    data_dir = Path(args.data_dir)
    schema = Schema.load(data_dir / "schema_resolved.json")
    rows = {split: read_jsonl(data_dir / f"{split}.jsonl") for split in ("train", "validation", "test")}
    start = time.time()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=args.max_features, sublinear_tf=True)
    features = {"train": vectorizer.fit_transform([r["text"] for r in rows["train"]])}
    for split in ("validation", "test"):
        features[split] = vectorizer.transform([r["text"] for r in rows[split]])
    labels = {split: np.asarray([r["label"] for r in rows[split]], dtype=np.int64) for split in rows}
    classifier = LogisticRegression(max_iter=1000, C=args.C, class_weight="balanced")
    classifier.fit(features["train"], labels["train"])

    results = {"name": "tfidf+logreg", "train_seconds": time.time() - start}
    for split, result_key in (("validation", "val"), ("test", "test")):
        probabilities = classifier.predict_proba(features[split])
        # sklearn omits a class if it never appears in the train split. Expand to the fixed schema order.
        logits = np.full((len(rows[split]), schema.n_classes), -20.0)
        logits[:, classifier.classes_] = np.log(np.clip(probabilities, 1e-12, 1.0))
        results[result_key] = evaluate(logits_to_probs(logits), labels[split])
    write_json(Path(args.out) / "metrics.json", results)
    print(f"TF-IDF test accuracy {results['test']['accuracy']:.4f}; macro-F1 {results['test']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
