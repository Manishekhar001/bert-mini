"""CPU latency and throughput: PyTorch vs ONNX vs INT8 (end to end, including tokenisation).

    python stage4/benchmark.py --dir artifacts/final
    python stage4/benchmark.py --dir artifacts/final --hf bert-base-uncased distilbert-base-uncased
Writes <dir>/bench.json
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from stage4.inference import Predictor, torch_logits_for_texts
from common.utils import read_jsonl, write_json


def measure(fn, texts, batch, warm=5, n_calls=100):
    if batch < 1:
        raise ValueError(f"batch must be positive, got {batch}")
    chunks = [texts[i:i + batch] for i in range(0, len(texts), batch)][:n_calls]
    if not chunks:
        raise ValueError("benchmark needs at least one report")
    for c in chunks[:warm]:
        fn(c)
    times = []
    for c in chunks:
        t0 = time.perf_counter()
        fn(c)
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.array(times)
    reports = sum(len(chunk) for chunk in chunks)
    return {"batch": batch, "median_ms_per_call": float(np.median(t)), "p95_ms_per_call": float(np.percentile(t, 95)),
            "ms_per_report": float(t.sum() / reports)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/final")
    ap.add_argument("--data_dir", default="data/asrs")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n_texts", type=int, default=320)
    ap.add_argument("--hf", nargs="*", default=[], help="HF encoders to time for comparison (random classifier)")
    args = ap.parse_args()

    d = Path(args.dir)
    texts = [r["text"] for r in read_jsonl(Path(args.data_dir) / "test.jsonl")[: args.n_texts]]
    results = {"threads": args.threads}

    for backend, fname in (("torch", "model.pt"), ("onnx", "model.onnx"), ("onnx_int8", "model.int8.onnx")):
        if not (d / fname).exists():
            print(f"skip {backend}: {fname} not found")
            continue
        p = Predictor(d, backend, num_threads=args.threads)
        size = (d / fname).stat().st_size / 1e6
        results[backend] = {"size_mb": size, "batch1": measure(p.logits, texts, 1),
                            "batch32": measure(p.logits, texts, 32, warm=1, n_calls=10)}
        print(f"{backend:<10} {size:6.1f} MB | batch 1: {results[backend]['batch1']['median_ms_per_call']:.1f} ms "
              f"(p95 {results[backend]['batch1']['p95_ms_per_call']:.1f}) | "
              f"batch 32: {results[backend]['batch32']['ms_per_report']:.2f} ms/report")

    if args.hf:
        import torch
        from transformers import AutoModel, AutoTokenizer

        from common.model import ReportClassifier
        from common.schema import Schema

        torch.set_num_threads(args.threads)
        schema = Schema.load(d / "schema_resolved.json")
        for name in args.hf:
            tok = AutoTokenizer.from_pretrained(name)
            m = ReportClassifier(AutoModel.from_pretrained(name).float(), schema.n_classes).eval()
            fn = lambda c: torch_logits_for_texts(m, tok, c, torch.device("cpu"), 256, 192, 32)  # noqa: E731
            results[f"hf:{name}"] = {"batch1": measure(fn, texts, 1), "batch32": measure(fn, texts, 32, warm=1, n_calls=10)}
            r = results[f"hf:{name}"]
            print(f"hf:{name:<24} batch 1: {r['batch1']['median_ms_per_call']:.1f} ms | "
                  f"batch 32: {r['batch32']['ms_per_report']:.2f} ms/report")

    write_json(d / "bench.json", results)
    print(f"\nsaved {d / 'bench.json'}")


if __name__ == "__main__":
    main()
