"""Upload the INT8 ONNX serving files into <repo>/onnx on the Hugging Face Hub.

    python stage4/push_onnx.py --repo yourname/mini-bert-asrs

Needs a login (`hf auth login`) or an HF_TOKEN (also read from a .env file in the project folder).
The Streamlit demo downloads only the onnx/ folder of this repo.
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
    ap.add_argument("--dir", default="artifacts/final_pretrained")
    ap.add_argument("--repo", required=True, help="name or owner/name")
    args = ap.parse_args()

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        username = api.whoami()["name"]
    except Exception as error:
        raise SystemExit(f"Not logged in to Hugging Face ({error}). Run `hf auth login`, or set HF_TOKEN.")
    owner, _, name = args.repo.rpartition("/")
    if owner.lower() in PLACEHOLDERS:
        raise SystemExit(f"'{owner}' is a placeholder, not a Hugging Face account. "
                         f"Use --repo {username}/{name}.")
    repo = f"{owner or username}/{name}"

    folder = Path(args.dir)
    needed = ["model.int8.onnx", "calibration.json", "schema_resolved.json", "tokenizer"]
    absent = [item for item in needed if not (folder / item).exists()]
    if absent:
        raise SystemExit(f"{folder} is missing {absent}. Run `python stage4/export_onnx.py --dir {folder}` first.")

    api.create_repo(repo, private=True, exist_ok=True)      # does not change an existing repo's visibility
    api.upload_folder(folder_path=str(folder), repo_id=repo, path_in_repo="onnx",
                      allow_patterns=["model.int8.onnx", "calibration.json", "schema_resolved.json", "tokenizer/*"])
    print(f"uploaded {folder} -> https://huggingface.co/{repo}/tree/main/onnx")


if __name__ == "__main__":
    main()
