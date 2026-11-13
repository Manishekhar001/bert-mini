"""Pack documents into fixed-length token blocks and store them as a Hugging Face Dataset"""

from __future__ import annotations

from pathlib import Path

class Packer :
    """Concatenate tokenized documents (separated by [SEP]) and cut them into [CLS] + body + [SEP] blocks"""

    def __init__(self, tokenizer, block_len : int = 256):
        if block_len < 3:
            raise ValueError(f"block len must be at least 3 ([CLS], one token, [SEP]) , got {block_len}")

        self.tok = tokenizer
        self.tok.model_max_length = 10**7
        self.cls, self.sep = tokenizer.cls_token_id, tokenizer.sep_token_id

        self.body = block_len - 2

        if self.cls is None or self.sep is None:
            raise ValueError("The tokenizer must define [CLS] and [SEP]token IDs")

        self.buf : list = []

    def add(self, texts) -> list:
        enc = self.tok(list(texts), add_special_tokens = False, truncation = False, verbose =False)["input_ids"]

        for ids in enc:
            self.buf.extend(ids)
            self.buf.append(self.sep)

        n = len(self.buf) // self.body

        blocks = [[self.cls] + self.buf[i * self.body : (i+1) * self.body] + [self.sep] for i in range(n)]

        self.buf = self.buf[n * self.body : ]
        return blocks


def blocks_from_docs(doc_iter , tokenizer , block_len : int = 256 , max_blocks : int | None = None, batch_docs: int = 256):
    """Stream documents -> packed [CLS] ... [SEP] blocks (lists of token ids)"""

    if batch_docs < 1:
        raise ValueError(f"batch_docs must be positive , got {batch_docs}")

    if max_blocks is not None and max_blocks < 1:
        raise ValueError(f"max_blocks must be positive when set, got {max_blocks}")

    packer, batch, n = Packer(tokenizer=tokenizer, block_len=block_len) , [] , 0
    for doc in doc_iter:
        if doc and doc.strip():
            batch.append(doc)
        if len(batch) >=batch_docs:
            for blk in packer.add(batch):
                yield blk
                n+=1
                if max_blocks is not None and n >= max_blocks:
                    return
            batch = []

    if batch:
        for blk in packer.add(batch):
            yield blk 
            n +=1
            if max_blocks is not None and n >= max_blocks:
                return

def _features(block_len : int):
    """
    Defines a helper function named _features. It takes the block length—for example, 256—and returns a description of the dataset’s columns and data types.

    The underscore at the start of the name is a Python convention meaning “this helper is mainly for use inside this file.”

    Imports two Hugging Face types:
    - Features describes the columns in the dataset.
    - Value describes the type of each individual value, such as an integer.
    They are imported inside the function because this helper needs them only when it runs.

    Builds and returns the dataset description.

    Read it from the inside out:
    - Value("int32") means each individual token ID is stored as a 32-bit integer.
    - List(..., length=block_len) means each row’s input_ids value is a list of exactly block_len integers.
    - {"input_ids": ...} says the dataset has a column named input_ids.
    - Features(...) wraps that description in Hugging Face’s dataset schema format.
    With the usual block_len=256, the meaning is:
    Each dataset row has one column called input_ids. Each cell in that column contains exactly 256 integer token IDs.

    int32 is large enough for BERT token IDs, while using less storage than a 64-bit integer.
    """
    from datasets import Features, Value

    try:
        from datasets import List
        return Features({"input_ids" :  List(Value("int32"), length=block_len)})
    except ImportError:
        from datasets import Sequence
        return Features({"input_ids": Sequence(Value("int32"), length=block_len)})


def save_blocks(blocks ,  out_dir, block_len : int = 256, chunk : int = 50_000):
    """Write blocks to a Hugging Face Dataset on disk (int32 ids, one fixed-length row per block).

    Written in chunks (no generator closure, so no hashing/caching surprises, and memory stays small), then joined.
    """

    import shutil

    if block_len < 3:
        raise ValueError(f"block_len must be at least 3, got {block_len}") 
    if chunk < 1:
        raise ValueError(f"chunk must be positive , got {chunk}")

    from datasets import Dataset , concatenate_datasets, load_from_disk

    out_dir = Path(out_dir)
    tmp = out_dir.parent / (out_dir.name + "_chunks")
    shutil.rmtree(tmp , ignore_errors=True)
    tmp.mkdir(parents=True)
    feats , parts , buf = _features(block_len=block_len) , [] ,[]

    def flush():
        if buf:
            path = tmp / f"c{len(parts):04d}"
            Dataset.from_dict({"input_ids"  :list(buf)} ,  features = feats).save_to_disk(str(path))
            parts.append(str(path))
            buf.clear()

    for b in blocks:
        buf.append([int(x) for x in b])
        if len(buf) >= chunk: 
            flush()

    flush()

    if not parts:
        raise RuntimeError("no blocks were produced; check the text source")

    ds = concatenate_datasets([load_from_disk(p) for p in parts])
    shutil.rmtree(out_dir, ignore_errors=True)
    ds.save_to_disk(str(out_dir))
    shutil.rmtree(tmp, ignore_errors=True)
    return load_from_disk(str(out_dir))