# Mini Jev — Stage 1 checkout

Mini Jev is a learning project for classifying aviation safety reports. This checkout contains the data-preparation utilities and Stage 1 language-model training code. It does not contain Stage 2, Stage 3, or Stage 4 training and serving scripts, so this guide documents only the code that is present.

## Environment

This checkout has a `pyproject.toml` and `uv.lock`; those are the dependency and Python-version source of truth. The project requires Python 3.13 or newer.

From this folder, install the locked environment and run project commands through `uv`:

```powershell
uv sync
uv run python -m pytest -q
```

Stage 1 is intended to use an NVIDIA GPU. Check whether PyTorch can see one:

```powershell
uv run python -c "import torch; print(torch.cuda.is_available())"
```

If this prints `False`, training will run on the CPU and will be very slow. Install a CUDA-enabled PyTorch build that matches your Windows and NVIDIA setup, then check again.

`requirement.txt` is a legacy pip-style dependency list. The `pyproject.toml` and `uv.lock` files are the source of truth for this checkout.

## Prepare the ASRS dataset

This optional step downloads the labeled ASRS reports and writes `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `schema_resolved.json` under `data/asrs/`:

```powershell
uv run python data_prep/asrs_prep.py
```

To inspect the token lengths of those reports:

```powershell
uv run python data_prep/token_stats.py --lengths 256 384 512
```

Stage 1 does not use ASRS data. Its training and validation text come from the general-English corpus below.

## Build the Stage 1 corpus

The default command streams English Wikipedia, tokenizes it, packs it into 256-token blocks, and saves the training and validation datasets under `data/text/wiki/`:

```powershell
uv run python stage1/build_corpus.py --out data/text/wiki
```

The defaults create about one million training blocks and 5,000 validation blocks. A smaller WikiText run is available with:

```powershell
uv run python stage1/build_corpus.py --name Salesforce/wikitext --config_name wikitext-103-raw-v1 --max_blocks 400000 --out data/text/wiki
```

## Train Stage 1

Stage 1 trains a randomly initialized four-layer BERT-mini on the general-English blocks. A frozen `bert-base-uncased` teacher guides it. The total loss is:

```text
MLM loss + 2 × teacher-logit KL loss + relation-KL loss
```

The model uses ordinary BERT self-attention internally. There is no separate loss comparing the teacher's and student's attention maps.

Run a small sanity check first:

```powershell
uv run python stage1/train.py --overfit
```

This uses a small slice of the corpus and 200 optimizer updates to check that the training pipeline runs. It is not the final model.

Run the configured training job:

```powershell
uv run python stage1/train.py
```

To continue from the newest checkpoint in `runs/stage1_distill/`:

```powershell
uv run python stage1/train.py --resume
```

The model and tokenizer are saved under `runs/stage1_distill/final/`. To run the plain-MLM control without a teacher:

```powershell
uv run python stage1/train.py --set objective=mlm --set out_dir=runs/stage1_mlm
```

Training settings are in `stage1/config.yaml`.

## Files in this checkout

```text
common/       shared model, loss, schema, and utility code
data_prep/    ASRS preparation and token-length inspection
stage1/       general-English corpus builder and distillation training
tests/        Stage 1 regression tests
data/         generated datasets and token blocks
runs/         generated checkpoints and logs
```