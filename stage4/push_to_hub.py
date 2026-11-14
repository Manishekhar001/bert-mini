"""Publish a trained model to the Hugging Face Hub (optional). Needs `huggingface-cli login` or HF_TOKEN.

    python stage4/push_to_hub.py --what encoder --path runs/stage2_domain/final --repo your-name/mini-jev-asrs-mlm
    python stage4/push_to_hub.py --what classifier --path artifacts/final --repo your-name/mini-jev-asrs

encoder    : the BERT-mini language model (stage 1 or 2 folder, standard save_pretrained format)
classifier : the packaged System-One model folder (model.pt, schema_resolved.json, calibration.json, tokenizer/, report.json)

Check the licences of the data you trained on before making a repo public (the default label dataset states none).
Repos are PRIVATE unless you pass --public.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["encoder", "classifier"], required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, private=not args.public, exist_ok=True)
    folder = Path(args.path)
    if not folder.is_dir():
        raise SystemExit(f"{folder} is not a folder")
    api.upload_folder(folder_path=str(folder), repo_id=args.repo,
                      ignore_patterns=["*.png"] if args.what == "encoder" else None)
    print(f"uploaded {folder} -> https://huggingface.co/{args.repo} ({'public' if args.public else 'private'})")
    if args.what == "encoder":
        print("load it with: BertForMaskedLM.from_pretrained('%s')" % args.repo)


if __name__ == "__main__":
    main()
