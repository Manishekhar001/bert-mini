"""Export one classifier output to ONNX and a dynamically quantized INT8 ONNX model."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from common.model import load_classifier
from common.utils import read_jsonl
from stage4.inference import Predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="artifacts/final")
    parser.add_argument("--data_dir", default="data/asrs")
    parser.add_argument("--n_check", type=int, default=300)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    if args.n_check < 0:
        parser.error("--n_check cannot be negative")

    from onnxruntime.quantization import QuantType, quantize_dynamic

    directory = Path(args.dir)
    model = load_classifier(directory / "model.pt", attn_impl="eager").eval()
    vocab_size = int(model.encoder.config.vocab_size)
    if vocab_size < 3:
        raise ValueError(f"model vocabulary must contain at least three token IDs, got {vocab_size}")
    sample_ids = torch.randint(0, vocab_size, (2, 64))
    sample_ids[:, 0], sample_ids[:, -1] = min(101, vocab_size - 1), min(102, vocab_size - 1)
    sample_attention = torch.ones_like(sample_ids)
    onnx_path, int8_path = directory / "model.onnx", directory / "model.int8.onnx"
    export_args = {
        "input_names": ["input_ids", "attention_mask"], "output_names": ["logits"],
        "opset_version": args.opset, "do_constant_folding": True,
        "dynamic_axes": {"input_ids": {0: "batch", 1: "sequence"},
                         "attention_mask": {0: "batch", 1: "sequence"}, "logits": {0: "batch"}},
    }
    with torch.no_grad():
        try:
            torch.onnx.export(model, (sample_ids, sample_attention), str(onnx_path), dynamo=False, **export_args)
        except TypeError:
            torch.onnx.export(model, (sample_ids, sample_attention), str(onnx_path), **export_args)
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB) and "
          f"{int8_path} ({int8_path.stat().st_size / 1e6:.1f} MB)")

    texts = [row["text"] for row in read_jsonl(Path(args.data_dir) / "test.jsonl")[:args.n_check]]
    reference = Predictor(directory, "torch").logits(texts)
    for backend in ("onnx", "onnx_int8"):
        actual = Predictor(directory, backend).logits(texts)
        agreement = float((reference.argmax(axis=1) == actual.argmax(axis=1)).mean()) if len(texts) else float("nan")
        max_diff = float(np.abs(reference - actual).max()) if len(texts) else float("nan")
        print(f"{backend}: same chosen class on {agreement:.1%}; "
              f"maximum absolute logit difference {max_diff:.4f}")


if __name__ == "__main__":
    main()
