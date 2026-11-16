"""Publish a trained model to the Hugging Face Hub. Needs a login (`hf auth login`) or an HF_TOKEN.

    python stage4/export_hf.py
    python stage4/push_to_hub.py --what hf_model --path artifacts/hf_model --repo mini-bert-asrs
    python stage4/push_to_hub.py --what encoder --path runs/stage2_domain/final --repo mini-jev-asrs-mlm

hf_model   : the standard Hugging Face model folder made by stage4/export_hf.py (config.json, model.safetensors,
             tokenizer, model code, model card). Load it with AutoModelForSequenceClassification (trust_remote_code).
encoder    : the BERT-mini language model (stage 1 or 2 folder, standard save_pretrained format)
classifier : the project's own package folder (model.pt, calibration.json, ...); kept for backup, not loadable
             with Transformers

--repo may be just a name (mini-bert-asrs): your Hugging Face username is looked up from your login and added.
A token is read from the environment, from the cached login, or from a .env file in the project folder.
Check the licences of the data you trained on before making a repo public. Repos are PRIVATE unless --public.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PLACEHOLDERS = {"your-name", "your-username", "username", "owner", "your_name"}


def load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines); existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["hf_model", "encoder", "classifier"], required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--repo", required=True, help="name or owner/name")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        username = api.whoami()["name"]
    except Exception as error:
        raise SystemExit(f"Not logged in to Hugging Face ({error}). Run `hf auth login`, or set HF_TOKEN "
                         "to a token with write access.")
    owner, _, name = args.repo.rpartition("/")
    if owner.lower() in PLACEHOLDERS:
        raise SystemExit(f"'{owner}' is a placeholder, not a Hugging Face account. Use --repo {name} "
                         f"(your username '{username}' is added automatically) or --repo {username}/{name}.")
    repo = f"{owner or username}/{name}"

    folder = Path(args.path)
    if not folder.is_dir():
        raise SystemExit(f"{folder} is not a folder")
    if args.what == "hf_model":
        needed = ["config.json", "model.safetensors", "modeling_minijev.py", "configuration_minijev.py"]
        absent = [item for item in needed if not (folder / item).exists()]
        if absent:
            raise SystemExit(f"{folder} is not a Hugging Face model folder (missing {absent}). "
                             "Run `python stage4/export_hf.py` first.")

    api.create_repo(repo, private=not args.public, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo,
                      ignore_patterns=["*.png", "__pycache__/*"] if args.what == "encoder" else ["__pycache__/*"])
    print(f"uploaded {folder} -> https://huggingface.co/{repo} ({'public' if args.public else 'private'})")
    if args.what == "hf_model":
        print("load it with: AutoModelForSequenceClassification.from_pretrained('%s', trust_remote_code=True)" % repo)
    elif args.what == "encoder":
        print("load it with: BertForMaskedLM.from_pretrained('%s')" % repo)


if __name__ == "__main__":
    main()
