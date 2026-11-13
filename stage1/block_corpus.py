"""Stage 1 data: stream general English text from the Hugging Face Hub, tokenise, pack into 256-token blocks,
and save as Hugging Face Datasets on disk.

    # default: Wikipedia, 1M blocks (about 256M tokens, about 1 GB on disk)
    python stage1/build_corpus.py --out data/text/wiki
    # smaller / faster start (WikiText-103, about 100M tokens)
    python stage1/build_corpus.py --name Salesforce/wikitext --config_name wikitext-103-raw-v1 \
        --max_blocks 400000 --out data/text/wiki

Writes <out>/train and <out>/validation. Stage 1 uses general text only; it does not load ASRS reports.
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1.text_blocks import blocks_from_docs, save_blocks


def hf_docs(name, config, split, text_col, seed, buffer):
    from datasets import load_dataset

    ds = load_dataset(name, config, split=split, streaming=True).shuffle(seed=seed, buffer_size=buffer)
    for row in ds:
        yield row[text_col]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="wikimedia/wikipedia")
    ap.add_argument("--config_name", default="20231101.en")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--out", default="data/text/wiki")
    ap.add_argument("--tokenizer", default="bert-base-uncased")
    ap.add_argument("--block_len", type=int, default=256)
    ap.add_argument("--max_blocks", type=int, default=1_000_000)
    ap.add_argument("--val_blocks", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle_buffer", type=int, default=10_000)
    args = ap.parse_args()
    if args.max_blocks < 1:
        ap.error("--max_blocks must be at least 1")
    if args.val_blocks < 1:
        ap.error("--val_blocks must be at least 1")

    from transformers import BertTokenizerFast

    tok = BertTokenizerFast.from_pretrained(args.tokenizer)
    out = Path(args.out)

    docs = hf_docs(args.name, args.config_name or None, args.split, args.text_col, args.seed, args.shuffle_buffer)
    # max_blocks counts training blocks; reserve a separate validation set from the same shuffled stream.
    blocks = blocks_from_docs(docs, tok, args.block_len, args.max_blocks + args.val_blocks)
    val = list(itertools.islice(blocks, args.val_blocks))
    """
    Takes the first args.val_blocks blocks from the stream and stores them as a list.
    - itertools.islice(blocks, 5000) takes at most the first 5,000 blocks.
    - list(...) consumes those blocks now and keeps them for saving as validation.
    After this, the blocks generator is paused at the point immediately following those validation blocks. When the script later continues consuming it, it supplies the remaining blocks.
    """

    save_blocks(val, out / "validation", args.block_len)
    train = save_blocks(blocks, out / "train", args.block_len)
    print(f"general text: {len(train):,} train blocks, {len(val):,} validation blocks -> {out}")


if __name__ == "__main__":
    main()
