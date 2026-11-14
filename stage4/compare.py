"""Collect metrics from classifier runs into one concise comparison table."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", default="artifacts/final")
    parser.add_argument("--out", default="results.md")
    args = parser.parse_args()

    runs = []
    for path in sorted(Path("runs").glob("*/metrics.json")):
        metrics = read_json(path)
        if "test" in metrics:
            runs.append(metrics)
    runs.sort(key=lambda item: item["test"].get("macro_f1", float("-inf")), reverse=True)
    lines = ["## Test-set results", "",
             "| model | parameters (M) | accuracy | macro-F1 | ECE |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for item in runs:
        test = item["test"]
        params = item.get("params")
        params = "" if params is None else f"{params / 1e6:.1f}"
        lines.append(f"| {item['name']} | {params} | {test['accuracy']:.3f} | "
                     f"{test['macro_f1']:.3f} | {test['ece']:.3f} |")

    benchmark = Path(args.final) / "bench.json"
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
