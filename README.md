# Mini Jev

Mini Jev is a learning project that classifies an aviation safety report into **one** category. It uses a compact BERT encoder and the Hugging Face dataset [`sookiemonster/asrs-narratives-rebalance`](https://huggingface.co/datasets/sookiemonster/asrs-narratives-rebalance).

The model does not answer a question written by the user. It reads a report and chooses one category from the dataset's fixed list of 18 categories.

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

## How the project runs

Run commands from the project folder, `D:\mini-jev`.

| Step | What it does | Main output |
| --- | --- | --- |
| Prepare data | Download reports, preserve the three splits, and save class names | `data/asrs/` |
| Stage 1 | Train a small encoder on general English, guided by a frozen BERT teacher | `runs/stage1_distill/final/` |
| Stage 2 | Continue language-model training on report text using LoRA | `runs/stage2_domain/final/` |
| Stage 3 | Train one report classifier from the integer labels | `runs/<run-name>/` |
| Stage 4 | Calibrate confidence, choose a review cutoff, and package inference files | `artifacts/final/` |

Stage 1 uses general English text only. Stage 2 uses report text without the labels. Stage 3 uses the labels. Stage 4 uses validation data to set probability calibration and review rules; the test split is used for reporting final results.

## 1. Install with uv

This project keeps its dependency list in `requirements.txt` and does not currently use a `pyproject.toml` or `uv.lock`. Use uv's virtual-environment and pip-compatible commands. Python 3.12 is recommended; the supported range is 3.10–3.12.

In PowerShell, open the project folder and create its `.venv` environment:

```powershell
uv python install 3.12
uv venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Run project commands from the project folder with `uv run`. This uses the local `.venv`; you do not need to activate it first. For example, `uv run python tools/check_env.py` runs the environment check inside `.venv`.

### If Windows blocks Python

`uv run` still has to start a Python interpreter. If Windows reports that an Application Control policy blocked `python.exe`, the command syntax is valid but the selected interpreter is not allowed by the device policy. Check the selected interpreter with `uv python find`. Use a Python installation approved for your device, or ask your administrator to approve the interpreter used by `.venv`; uv cannot override Application Control.

For NVIDIA training, choose the Windows, Pip, Python, and CUDA options in the [PyTorch installation selector](https://pytorch.org/get-started/locally/). Use its command to install `torch` into this environment (replace its `pip install` prefix with `uv pip install --python .venv\Scripts\python.exe`), then run the requirements-install command above. The `torch` requirement accepts versions from 2.3 up to, but not including, 3.0.

Check the environment and run the tests:

```powershell
uv run python tools/check_env.py
uv run python -m pytest -q
```

For uv's command reference, see the [uv pip interface](https://docs.astral.sh/uv/pip/).

## 2. Prepare the labeled reports

This downloads the Hugging Face dataset and writes `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `schema_resolved.json` under `data/asrs/`.

```powershell
uv run python data_prep/prepare_asrs.py
```

To see report token lengths and how much a chosen input window would truncate:

```powershell
uv run python data_prep/token_stats.py --lengths 256 384 512
```

## 3. Stage 1: learn general language patterns

Build the general-English token blocks. The default downloads Wikipedia and writes one million 256-token training blocks plus a validation set. The optional ASRS validation blocks are for monitoring only; they are not used to train the model.

```powershell
uv run python stage1/build_corpus.py --out data/text/wiki --asrs_val_dir data/asrs
```

Train the small model with masked-language modeling and teacher guidance:

```powershell
uv run python stage1/train.py
```

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

The TF-IDF baseline is optional and does not require the BERT stages:

```powershell
uv run python stage3/baselines.py
```

## 6. Stage 4: calibrate and package

Temperature calibration and the confidence cutoff are fitted on validation reports. A prediction below the cutoff is marked for human review. By default, Stage 4 also compares confidence and energy scores against outside-domain Wikipedia text, keeps whichever score better separates the two validation samples, and saves a low-score boundary. This is a warning based on model scores, not proof that a report is or is not about aviation.

```powershell
uv run python stage4/calibrate.py --run runs/C_distill_domain --out artifacts/final
```

To skip the optional outside-domain download and warning:

```powershell
uv run python stage4/calibrate.py --run runs/C_distill_domain --out artifacts/final --no_ood
```

The package contains `model.pt`, the tokenizer, the class-name mapping, calibration settings, and a report of validation/test metrics. Test results are for evaluation; they are not used to fit the calibration settings.

Optional: export regular and INT8 ONNX models and compare their predictions with PyTorch:

```powershell
uv run python stage4/export_onnx.py --dir artifacts/final
```

Optional: benchmark inference speed or collect run metrics:

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

The service looks for `artifacts/final` by default. To select another package or backend in PowerShell, set `MINIJEV_DIR` or `MINIJEV_BACKEND` before starting it. Backends are `torch`, `onnx`, and `onnx_int8`.

## Folder guide

```text
common/       shared schema, model, losses, metrics, and utilities
data_prep/    download and normalize the Hugging Face dataset
stage1/       general-English corpus builder and language-model training
stage2/       aviation-language adaptation
stage3/       one-label classifier and TF-IDF baseline
stage4/       calibration, inference, export, API, and demo
tools/        environment checker
tests/        project tests
data/         generated prepared data and text blocks
runs/         generated training checkpoints and metrics
artifacts/    generated packaged models
```

This is a learning project, not a certified aviation safety system. Do not use its predictions as the sole basis for an operational safety decision.
