"""Download and normalize the project's single-label Hugging Face dataset.

Run from the project root:
    python data_prep/asrs_prep.py

Writes train/validation/test JSONL files and the label-id-to-name mapping under data/asrs/.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.schema import Schema
from common.utils import write_jsonl


DATASET_ID = "sookiemonster/asrs-narratives-rebalance"
SPLITS = ("train", "validation", "test")


def clean_report(text: str) -> str:
    """Remove the dataset's optional `Narrative 1 - '...'` wrapper."""
    text = str(text or "").strip()
    text = re.sub(r"^Narrative\s+\d+\s*-\s*", "", text, count=1, flags=re.IGNORECASE)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1].strip()
    return text


def class_names_from_feature(feature) -> list[str]:
    """Read the ordered class names from Hugging Face's ClassLabel metadata."""
    names = getattr(feature, "names", None)
    if not names:
        raise ValueError("The dataset's `label` column has no ClassLabel names; cannot map label IDs safely.")
    return [str(name) for name in names]


def convert_split(dataset_split, split_name: str, class_names: list[str]) -> list[dict]:
    rows = []
    for row_number, row in enumerate(dataset_split):
        label = int(row["label"])
        if not 0 <= label < len(class_names):
            raise ValueError(f"{split_name} row {row_number} has label {label}, outside 0..{len(class_names) - 1}")
        text = clean_report(row["text"])
        if not text:
            continue
        rows.append({"id": str(row["acn"]), "text": text, "label": label})
    return rows


def prepare_dataset(dataset, out_dir: str | Path = "data/asrs") -> Schema:
    """Convert an already-loaded HF DatasetDict. Kept separate so the conversion is easy to test."""
    missing = [split for split in SPLITS if split not in dataset]
    if missing:
        raise ValueError(f"Dataset is missing required split(s): {', '.join(missing)}")
    required_columns = {"acn", "text", "label"}
    for split in SPLITS:
        absent = required_columns - set(dataset[split].column_names)
        if absent:
            raise ValueError(
                f"{split} split is missing required column(s): {', '.join(sorted(absent))}; "
                "expected acn, text and label"
            )
    class_names = class_names_from_feature(dataset["train"].features["label"])
    for split in SPLITS[1:]:
        split_names = class_names_from_feature(dataset[split].features["label"])
        if split_names != class_names:
            raise ValueError(f"{split} split uses a different label-to-name mapping than train")
    converted = {split: convert_split(dataset[split], split, class_names) for split in SPLITS}
    counts = Counter(row["label"] for row in converted["train"])
    schema = Schema(
        classes=class_names,
        counts=[int(counts[i]) for i in range(len(class_names))],
        total_samples=len(converted["train"]),
        dataset=DATASET_ID,
    )
    out = Path(out_dir)
    for split in SPLITS:
        write_jsonl(out / f"{split}.jsonl", converted[split])
        print(f"wrote {len(converted[split]):,} {split} reports -> {out / (split + '.jsonl')}")
    schema.save(out / "schema_resolved.json")
    print(f"saved {schema.n_classes} class names -> {out / 'schema_resolved.json'}")
    for i, (name, count) in enumerate(zip(schema.classes, schema.counts)):
        print(f"  {i:2d} {name:<42} {count:>7,}")
    return schema


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare sookiemonster/asrs-narratives-rebalance")
    parser.add_argument("--out_dir", default="data/asrs")
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading {DATASET_ID} from Hugging Face...")
    dataset = load_dataset(DATASET_ID)
    prepare_dataset(dataset, args.out_dir)


if __name__ == "__main__":
    main()
