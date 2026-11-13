"""How much of the ASRS data fits in the context window? Run after data_prep/asrs_prep.py.

    python data_prep/token_stats.py
    python data_prep/token_stats.py --lengths 256 384 512
Prints token-length percentiles of the narratives and the share of reports longer than each window.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.utils import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/asrs")
    ap.add_argument("--tokenizer", default="bert-base-uncased")
    ap.add_argument("--lengths", type=int, nargs="*", default=[256, 384, 512])
    args = ap.parse_args()

    from transformers import BertTokenizerFast

    tok = BertTokenizerFast.from_pretrained(args.tokenizer)
    tok.model_max_length = 10 ** 7
    rows = read_jsonl(Path(args.data_dir) / "train.jsonl")
    if not rows:
        raise SystemExit("No training reports found. Run python data_prep/asrs_prep.py first.")
    if any(length < 2 for length in args.lengths):
        raise SystemExit("Every requested length must leave room for [CLS] and [SEP] (minimum 2).")
    texts = [r["text"] for r in rows]
    n = np.array([len(x) for x in tok(texts, add_special_tokens=False, truncation=False)["input_ids"]]) + 2   # + CLS, SEP
    print(f"{len(n):,} train reports, tokens per report (including [CLS] and [SEP])")
    print(f"  mean {n.mean():.0f} | median {np.median(n):.0f} | p90 {np.percentile(n, 90):.0f} | "
          f"p95 {np.percentile(n, 95):.0f} | max {n.max()}")
    for L in args.lengths:
        over = (n > L).mean()
        lost = np.maximum(n - L, 0).sum() / n.sum()
        print(f"  window {L}: {over:.1%} of reports are longer; {lost:.1%} of all tokens would be cut without head+tail")


if __name__ == "__main__":
    main()
