# bert-mini

bert-mini classifies an aviation safety report into **one** category. It uses a compact BERT encoder and the Hugging Face dataset [`sookiemonster/asrs-narratives-rebalance`](https://huggingface.co/datasets/sookiemonster/asrs-narratives-rebalance).

The model does not answer a question written by the user. It reads a report and chooses one category from the dataset's fixed list of 18 categories.

## Current results

On the held-out test split (9,868 reports; always guessing the most common class scores 0.342 accuracy), taken from `results.md`. The best bert-mini run is `D_full_ft`: Stage 1 distillation, Stage 2 domain adaptation, Stage 3 full-encoder fine-tuning (`use_lora=false`), then Stage 4 prior correction and temperature scaling.

| Model | Params | Accuracy | Macro-F1 | Calibration error (ECE) |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + logistic regression (raw) | - | 0.600 | 0.313 | 0.106 |
| bert-mini `D_full_ft` (raw) | 11.2M | 0.591 | 0.288 | 0.044 |
| **bert-mini `D_full_ft` + prior correction + temperature scaling** | 11.2M | **0.638** | **0.299** | **0.014** |
| Reference: Google `bert_uncased_L-4_H-256_A-4`, same Stage 3 full fine-tune + prior correction | 11.2M | 0.638 | 0.294 | 0.017 |
| Reference: bert-mini `C_distill_domain` (LoRA in Stage 3) + prior correction | 11.2M | 0.604 | 0.268 | 0.021 |

How to read this:

- After correction, `D_full_ft` matches a fine-tuned off-the-shelf BERT-mini of the same size and has the lowest calibration error of all runs. The macro-F1 gap to the Google model (0.005) is within single-seed noise, so this is a tie, not a win. Before correction, the Google model is slightly ahead (0.597 accuracy, 0.295 macro-F1).
- It does **not** beat TF-IDF on macro-F1 (0.299 vs 0.313), which is dominated by categories with few examples. TF-IDF was not given the prior correction, so its accuracy is not a like-for-like comparison with the corrected rows.
- Full-encoder fine-tuning beat LoRA in Stage 3 (0.299 vs 0.268 macro-F1) from the same Stage 2 encoder.
- Each result is from one training run (one seed). `results.md` has the full table of runs, a per-class table, a confusion matrix, and CPU latency.

Distillation is used **only in Stage 1 (pretraining)**; Stages 2, 3 and 4 never use a teacher.

## Dataset format

The source dataset has one row per report:

| Field | Meaning |
| --- | --- |
| `acn` | Report identifier |
| `text` | Report narrative |
| `label` | One integer category ID |

The `label` column is an integer with class names stored in Hugging Face dataset metadata. Preparation copies those names into `schema_resolved.json`, so the same ID maps to the same name during training and prediction.

Prepared data uses this format:

```json
{"id":"2097155","text":"During pushback ...","label":14}
```

The dataset viewer shows this example with label ID `14`; the actual category name is read from its class metadata, not hard-coded in the project.

### The class mix differs between splits

The training split is rebalanced, but the validation and test splits keep the natural mix of categories. For example, `humanfactors` is 24% of training reports but 34% of test reports, `aircraft` is 15% vs 34%, and `procedure` is 25% vs 10%. A model trained on the rebalanced data therefore has class frequencies that do not match the data it is used on. Stage 4 corrects this with a per-class offset fitted on validation reports (see section 6), and `results.md` reports results both with and without that correction.

## How the project runs

Run commands from the project's root folder.

| Step | What it does | Main output |
| --- | --- | --- |
| Prepare data | Download reports, preserve the three splits, and save class names | `data/asrs/` |
| Stage 1 | Train a small encoder on general English, guided by a frozen BERT teacher | `runs/stage1_distill/final/` |
| Stage 2 | Continue language-model training on report text using LoRA | `runs/stage2_domain/final/` |
| Stage 3 | Train one report classifier from the integer labels | `runs/<run-name>/` |
| Stage 4 | Correct for the class-mix difference, calibrate confidence, choose a review cutoff, and package inference files | `artifacts/final/` |
| Export | Save a standard Hugging Face model folder and optionally publish it | `artifacts/hf_model/` |

Stage 1 uses general English text only. Stage 2 uses report text without the labels. Stage 3 uses the labels. Stage 4 uses validation data to set the class-mix correction, probability calibration, and review rules; the test split is used only for reporting final results.

## 1. Install with uv

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`; the Python version (3.12) is set in `.python-version`. Python 3.12 is recommended; the supported range is 3.10–3.12.

In PowerShell, open the project folder and create its `.venv` environment:

```powershell
uv python install 3.12
uv sync
```

Run project commands from the project folder with `uv run`. This uses the local `.venv`; you do not need to activate it first. For example, `uv run python -m pytest -q` runs the tests inside `.venv`.

### If Windows blocks Python

`uv run` still has to start a Python interpreter. If Windows reports that an Application Control policy blocked `python.exe`, the command syntax is valid but the selected interpreter is not allowed by the device policy. Check the selected interpreter with `uv python find`. Use a Python installation approved for your device, or ask your administrator to approve the interpreter used by `.venv`; uv cannot override Application Control.

`pyproject.toml` directs uv to take `torch` from PyTorch's official CUDA 12.6 wheel index. This installs a CUDA-enabled PyTorch build for the Windows/NVIDIA setup, while PyPI supplies the other dependencies. The `torch` requirement accepts versions from 2.3 up to, but not including, 3.0. For a different operating system or CUDA version, choose the matching option in the [PyTorch installation selector](https://pytorch.org/get-started/locally/) and change the index URL under `[[tool.uv.index]]` in `pyproject.toml`.

After installation, check that PyTorch can see the GPU:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

On a working CUDA setup, this should print a CUDA build version and `True` for CUDA availability. If it prints `None` and `False`, the environment still has a CPU-only PyTorch build.

Run the tests:

```powershell
uv run python -m pytest -q
```

For uv's command reference, see the [uv documentation](https://docs.astral.sh/uv/).

## 2. Prepare the labeled reports

This downloads the Hugging Face dataset and writes `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `schema_resolved.json` under `data/asrs/`.

```powershell
uv run python data_prep/asrs_prep.py
```

To see report token lengths and how much a chosen input window would truncate:

```powershell
uv run python data_prep/token_stats.py --lengths 256 384 512
```

## 3. Stage 1: learn general language patterns

Build the general-English token blocks. This downloads Wikipedia and writes one million 256-token training blocks plus a validation set. Stage 1 loads only this general-English corpus; ASRS reports are first used in Stage 2.

```powershell
uv run python stage1/build_corpus.py --out data/text/wiki
```

Train the small model with masked-language modeling and teacher guidance:

```powershell
uv run python stage1/train.py
```

Stage 1 combines masked-token prediction loss, teacher logit-KL loss, and relation-KL loss between matched student and teacher layers. BERT's normal self-attention remains part of every encoder layer; the project does not add a separate loss that matches the teacher's attention maps.

**Distillation happens only here.** Stage 1 is the only stage that uses the teacher; Stages 2–4 train the student on its own.

**How close is the student to the teacher?** Stage 1 logs the masked-token accuracy of the student (`general_acc`) and of the teacher (`general_t_acc`) on held-out Wikipedia blocks. The teacher scores 61.6%. The student reached 45.2% after the first epoch (73% of the teacher) and 46.6% after the second (76%), and was still improving slowly. This is a different measure from the "95% of the teacher" figures reported for DistilBERT-style models, which are scores on downstream tasks, usually with larger students. Stage 2 reads `runs/stage1_distill/final/` by default (see `stage2/config.yaml`); if you trained a second epoch in `runs/stage1_distill_epoch2/`, point Stage 2 at it.

The standard output is `runs/stage1_distill/final/`. Add `--resume` to continue from the latest checkpoint in that run folder. To do a short pipeline sanity check, use `uv run python stage1/train.py --overfit`; this deliberately trains on only a few blocks and is not a useful final model.

## 4. Stage 2: adapt to aviation report language

Stage 2 continues from Stage 1. It learns from the report text without looking at the category labels. By default, query/value LoRA adapters and the position-embedding table are trained; the other pretrained weights stay fixed. The input window grows to 512 tokens, and long reports keep an opening portion and an ending portion.

```powershell
uv run python stage2/train.py
```

The standard output is `runs/stage2_domain/final/`. Stage 2 uses training reports for learning and validation reports for monitoring; it does not train on test reports. The final LoRA adapters are merged into the ordinary BERT weights. Use `--resume` to continue from the latest Trainer checkpoint.

## 5. Stage 3: learn the report categories

Stage 3 adds one shared classifier to the encoder. For each report it produces 18 scores, one for each category. Softmax converts these scores to probabilities that add up to 100%, and cross-entropy compares them with that report's single integer label.

For a pretrained run, Stage 3 trains the query/value LoRA adapters, the attention-pooling layer, and the output classifier. The pretrained encoder's other weights remain fixed. A `random` initialization run automatically turns LoRA off and trains the full encoder, so the random baseline is not a frozen random model.

```powershell
uv run python stage3/train.py --run C_distill_domain --init runs/stage2_domain/final
```

The run saves `best.pt`, validation and test logits, the class mapping, and metrics under `runs/C_distill_domain/`. Training settings are in `stage3/config.yaml`.

If training runs out of GPU memory, use a smaller per-step batch and increase gradient accumulation to keep their product similar. For example, these two overrides keep the effective batch at 64:

```powershell
uv run python stage3/train.py --run C_distill_domain --init runs/stage2_domain/final --set args.per_device_train_batch_size=4 --set args.gradient_accumulation_steps=16
```

LoRA is the default and stays available. To fine-tune the whole encoder instead (more trainable parameters, often a stronger classifier), add `--set use_lora=false` and use a smaller encoder learning rate. This does not use a teacher.

```powershell
uv run python stage3/train.py --run D_full_ft --init runs/stage2_domain/final --set use_lora=false --set args.learning_rate=1e-4
```

For a reference point you can fine-tune the full `bert-base-uncased` model on the same reports. It is only a yardstick for how well a big model does on this task; it is never used to train the small model.

```powershell
uv run python stage3/train.py --run T_bert_base --init hf:bert-base-uncased --set use_lora=false --set args.learning_rate=3e-5 --set args.per_device_train_batch_size=4 --set args.gradient_accumulation_steps=16 --set args.num_train_epochs=6
```

The TF-IDF baseline is optional, does not require the BERT stages, and saves its scores so `stage4/compare.py` can treat it like any other run:

```powershell
uv run python stage3/baselines.py
```

## 6. Stage 4: calibrate and package

First, Stage 4 fits a **per-class prior correction**: one adder per category, fitted on validation reports, that moves the model's class frequencies from the rebalanced training mix toward the natural mix. How much of the correction to apply (0 to 100%) is chosen by cross-validation inside the validation set, so the test split stays untouched. The offsets are saved as `class_bias` in `calibration.json` and applied automatically by `stage4/inference.py`. Add `--no_bias` to skip this step.

Then temperature calibration and the confidence cutoff are fitted on validation reports. A prediction below the cutoff is marked for human review.

```powershell
uv run python stage4/calibrate.py --run runs/C_distill_domain --out artifacts/final
```

The package contains `model.pt`, the tokenizer, the class-name mapping, calibration settings (including the prior-correction offsets), and a report of validation/test metrics. Test results are for evaluation; they are not used to fit the calibration settings.

Optional: export regular and INT8 ONNX models and compare their predictions with PyTorch:

```powershell
uv run python stage4/export_onnx.py --dir artifacts/final
```

Optional: benchmark inference speed, then write `results.md` (raw and prior-corrected results for every run, a per-class table, a confusion matrix, and latency):

```powershell
uv run python stage4/benchmark.py --dir artifacts/final
uv run python stage4/compare.py
```

## 7. Run the model

The API and demo use the INT8 ONNX model by default, so export it first. Start the API:

```powershell
uv run uvicorn stage4.serve.app:app --host 127.0.0.1 --port 8000
```

In another PowerShell window, send one report:

```powershell
$body = @{ texts = @("On final approach, the crew received a traffic alert and discontinued the approach.") } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $body
```

The response includes the chosen category, its confidence, probabilities for all categories, and whether the report should be reviewed by a person. The demo provides the same prediction through a small web page:

```powershell
uv run streamlit run stage4/demo/streamlit_app.py
```

The service looks for `artifacts/final_pretrained` by default. To select another package or backend in PowerShell, set `BERTMINI_DIR` or `BERTMINI_BACKEND` before starting it. Backends are `torch`, `onnx`, and `onnx_int8`.

## 8. Export to Hugging Face format and publish

Stage 4's `model.pt` is the project's own format. To get a standard Hugging Face model folder (`config.json`, `model.safetensors`, tokenizer, model card), run:

```powershell
uv run python stage4/export_hf.py
```

This writes `artifacts/hf_model/` and then reloads it the way a Hub user would, checking that its answers match the packaged model. The classifier has an attention-pooling head, so it is not a stock `BertForSequenceClassification`. The folder therefore includes two small Python files (`configuration_bertmini.py`, `modeling_bertmini.py`) that define the model, and loading requires `trust_remote_code=True`. The calibration settings travel inside `config.json`, and `model.classify(...)` applies them.

Publish it (repos are private unless you add `--public`):

```powershell
uv run huggingface-cli login
uv run python stage4/push_to_hub.py --what hf_model --path artifacts/hf_model --repo mini-bert-asrs
```

`--repo` may be just a name; your Hugging Face username is looked up from your login. On newer `huggingface_hub` versions the login command is `hf auth login`. You can also set `HF_TOKEN` to a token with write access; the script also reads a `.env` file in the project folder. Check the licence of the training data before making the repo public. Other `--what` choices are `encoder` (the Stage 1 or 2 language model, a standard BERT folder) and `classifier` (the project package, kept as a backup).

Anyone can then use it like this (replace `your-username` with the account that owns the repo):

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "your-username/mini-bert-asrs"
model = AutoModelForSequenceClassification.from_pretrained(repo, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
print(model.classify(["On final approach, the crew received a traffic alert."], tokenizer)[0]["label"])
```

## Folder guide

```text
common/       shared schema, model, losses, metrics, and utilities
data_prep/    download and normalize the Hugging Face dataset
stage1/       general-English corpus builder and language-model training
stage2/       aviation-language adaptation
stage3/       one-label classifier and TF-IDF baseline
stage4/       calibration, inference, ONNX and Hugging Face export, API, and demo
tests/        project tests
data/         generated prepared data and text blocks
runs/         generated training checkpoints and metrics
artifacts/    generated packaged models
```

This is not a certified aviation safety system. Do not use its predictions as the sole basis for an operational safety decision.
