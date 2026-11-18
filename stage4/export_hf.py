"""Export the packaged classifier (artifacts/final) as a standard Hugging Face model folder.

    python stage4/export_hf.py                         # -> artifacts/hf_model
    python stage4/push_to_hub.py --what hf_model --path artifacts/hf_model --repo mini-bert-asrs

The folder holds config.json, model.safetensors, the tokenizer, the two small Python files that define the model
class (loaded with trust_remote_code=True) and a model card (README.md). The model has an attention-pooling head, so
it is not a stock BertForSequenceClassification; the custom class keeps the exact architecture that was trained.
After exporting, the script loads the folder back through AutoModelForSequenceClassification and checks that it
gives the same answers as the packaged model.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CODE_FILES = ("configuration_bertmini.py", "modeling_bertmini.py")

CARD = """---
language: en
library_name: transformers
pipeline_tag: text-classification
tags:
- text-classification
- aviation
- safety-reports
- asrs
- distillation
- bert
---

# {name}

A small ({params_m:.1f}M parameter) BERT-style classifier that assigns one of {n_classes} categories to an aviation
safety report narrative. It is built in three steps: general knowledge distillation from `bert-base-uncased`,
domain adaptation on ASRS text, then supervised classification.

## How to use

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "{repo_hint}"
model = AutoModelForSequenceClassification.from_pretrained(repo, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

result = model.classify(["The crew was cleared for the visual approach but ..."], tokenizer)[0]
print(result["label"], result["confidence"], result["needs_review"])
```

`trust_remote_code=True` is required because the model uses attention pooling, which is not part of the stock BERT
classifier. The code is the two small files `configuration_bertmini.py` and `modeling_bertmini.py` in this repository;
read them before running.

`model.classify(...)` applies the same steps as the original project: keeps the first {head_tokens} and the last
tokens of long reports (window of {max_len} tokens), adds a per-class prior correction fitted on validation data,
applies temperature scaling, and flags a report for human review when confidence is below
{threshold_text}. Calling `model(input_ids, attention_mask)` directly returns raw, uncorrected scores.

## Classes

{class_list}

## Training pipeline

1. **Stage 1, distillation.** A random-initialised BERT-mini (4 layers, hidden size 256, 4 heads) learns from a frozen
   `bert-base-uncased` teacher on Wikipedia text (masked-token loss, teacher soft labels, hidden-relation matching).
2. **Stage 2, domain adaptation.** Masked-language-model training on ASRS narratives (positions extended to 512).
3. **Stage 3, classifier.** {stage3_text}

Training data: `{dataset}`. Check its licence before relying on or redistributing this model.

## Results (held-out test split, {test_n} reports)

{results_text}

## Limitations

- Accuracy is modest and macro-F1 is low: categories with few examples are rarely predicted correctly, and some
  categories overlap in meaning. In the project's own comparison a simple TF-IDF + logistic-regression baseline
  reached similar or higher raw accuracy than this model, so treat it as a triage aid, not an authority.
- Calibration helps: when the model says it is confident, it is closer to being right. The review flag is meant to
  route uncertain reports to a person.
- Not for operational or safety-critical decisions.
"""


def results_block(report: dict, bias: dict, test_n) -> str:
    after = report.get("after_calibration", {})
    if not after:
        return "Run `stage4/calibrate.py` to produce metrics for this package."
    lines = ["| accuracy | macro-F1 | expected calibration error |", "| ---: | ---: | ---: |",
             f"| {after['accuracy']:.3f} | {after['macro_f1']:.3f} | {after['ece']:.3f} |"]
    review = report.get("review", {})
    if review.get("test_coverage") is not None and review.get("test_accuracy_when_accepted") is not None:
        lines.append("")
        lines.append(f"Automatic acceptance at the review cutoff: {review['test_coverage']:.1%} of reports, "
                     f"{review['test_accuracy_when_accepted']:.1%} correct among them.")
    if bias and bias.get("applied"):
        lines.append("")
        lines.append(f"The per-class prior correction (strength {bias['strength']:.2f}) moved test accuracy from "
                     f"{bias['before']['accuracy']:.3f} to {bias['after']['accuracy']:.3f} and macro-F1 from "
                     f"{bias['before']['macro_f1']:.3f} to {bias['after']['macro_f1']:.3f}.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="artifacts/final", help="package made by stage4/calibrate.py")
    parser.add_argument("--out", default="artifacts/hf_model")
    parser.add_argument("--data_dir", default="data/asrs")
    parser.add_argument("--name", default="mini-bert-asrs", help="title used in the model card")
    parser.add_argument("--repo_hint", default=None, help="owner/name shown in the card's usage example")
    parser.add_argument("--n_check", type=int, default=64, help="test reports used to verify the exported model")
    args = parser.parse_args()

    from transformers import AutoModelForSequenceClassification, BertTokenizerFast

    from common.model import load_classifier
    from common.schema import Schema
    from common.utils import read_json, read_jsonl
    from stage4.hf.configuration_bertmini import BertMiniConfig
    from stage4.hf.modeling_bertmini import BertMiniForSequenceClassification
    from stage4.inference import Predictor

    src, out = Path(args.dir), Path(args.out)
    schema = Schema.load(src / "schema_resolved.json")
    cal = read_json(src / "calibration.json")
    report = read_json(src / "report.json") if (src / "report.json").exists() else {}
    source = load_classifier(src / "model.pt").eval()
    enc = source.encoder.config
    data_cfg = cal.get("data_cfg", {})

    config = BertMiniConfig(
        vocab_size=enc.vocab_size, hidden_size=enc.hidden_size, num_hidden_layers=enc.num_hidden_layers,
        num_attention_heads=enc.num_attention_heads, intermediate_size=enc.intermediate_size,
        hidden_act=enc.hidden_act, hidden_dropout_prob=enc.hidden_dropout_prob,
        attention_probs_dropout_prob=enc.attention_probs_dropout_prob,
        max_position_embeddings=enc.max_position_embeddings, type_vocab_size=enc.type_vocab_size,
        initializer_range=enc.initializer_range, layer_norm_eps=enc.layer_norm_eps, pad_token_id=enc.pad_token_id,
        pooling=source.pooling, report_max_len=data_cfg.get("max_len", 512),
        report_head_tokens=data_cfg.get("head_tokens", 384),
        calibration_temperature=cal.get("temperature", 1.0), class_bias=cal.get("class_bias"),
        confidence_threshold=cal.get("confidence_threshold"),
        id2label=dict(enumerate(schema.classes)), label2id={name: i for i, name in enumerate(schema.classes)},
        architectures=["BertMiniForSequenceClassification"],
    )
    config.auto_map = {"AutoConfig": "configuration_bertmini.BertMiniConfig",
                       "AutoModelForSequenceClassification": "modeling_bertmini.BertMiniForSequenceClassification"}
    model = BertMiniForSequenceClassification(config)
    missing, unexpected = model.load_state_dict(source.state_dict(), strict=False)
    unexpected = [key for key in unexpected if not key.endswith("position_ids")]
    if missing or unexpected:
        raise SystemExit(f"weights do not match the Hugging Face class: missing {missing}, unexpected {unexpected}")

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    BertTokenizerFast.from_pretrained(str(src / "tokenizer")).save_pretrained(str(out))
    for name in CODE_FILES:
        shutil.copy(Path(__file__).parent / "hf" / name, out / name)
    shutil.copy(src / "calibration.json", out / "calibration.json")

    run_folder = Path(report["run"]) if report.get("run") else None
    training = {}
    if run_folder and (run_folder / "metrics.json").exists():
        training = read_json(run_folder / "metrics.json").get("training", {})
    stage3_text = (f"Classifier with attention pooling, trained with {training['strategy']}."
                   if training.get("strategy") else "Classifier with attention pooling.")
    cutoff = cal.get("confidence_threshold")
    test_n = len(read_jsonl(Path(args.data_dir) / "test.jsonl")) if (Path(args.data_dir) / "test.jsonl").exists() else "?"
    params = sum(p.numel() for p in model.parameters())
    card = CARD.format(
        name=args.name, params_m=params / 1e6, n_classes=schema.n_classes,
        repo_hint=args.repo_hint or "your-username/" + args.name, head_tokens=config.report_head_tokens,
        max_len=config.report_max_len,
        threshold_text=f"{cutoff:.3f}" if cutoff is not None else "its calibrated cutoff",
        class_list="\n".join(f"- {name}" for name in schema.classes),
        stage3_text=stage3_text, dataset=schema.dataset or "see the project README", test_n=test_n,
        results_text=results_block(report, report.get("class_bias", {}), test_n),
    )
    (out / "README.md").write_text(card, encoding="utf-8")
    print(f"wrote Hugging Face model folder -> {out} ({params / 1e6:.1f}M parameters)")

    # Verify: reload through the Auto classes exactly like a user of the Hub repo would, and compare answers.
    texts = [row["text"] for row in read_jsonl(Path(args.data_dir) / "test.jsonl")[:max(args.n_check, 0)]]
    if texts:
        loaded = AutoModelForSequenceClassification.from_pretrained(str(out), trust_remote_code=True).eval()
        tokenizer = BertTokenizerFast.from_pretrained(str(out))
        got = loaded.classify(texts, tokenizer)
        want = Predictor(src, "torch").predict(texts)
        same = sum(a["label_id"] == b["label_id"] for a, b in zip(got, want)) / len(texts)
        gap = max(abs(a["confidence"] - b["confidence"]) for a, b in zip(got, want))
        flags = sum(a["needs_review"] == b["needs_review"] for a, b in zip(got, want)) / len(texts)
        print(f"check on {len(texts)} test reports: same class {same:.1%}, same review flag {flags:.1%}, "
              f"largest confidence difference {gap:.2e}")
        if same < 0.98:
            raise SystemExit("the exported model disagrees with the packaged model; do not publish it")


if __name__ == "__main__":
    main()
