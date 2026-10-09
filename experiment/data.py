"""Datasets, tokenisation and per-worker shards.

* ``shakespeare``: TinyShakespeare at character level (the smoke test).
* ``tinystories``: TinyStories (V2, GPT-4 stories) with a byte-level BPE tokeniser trained here.
  Only the first ``TS_TRAIN_BYTES`` of the training file are downloaded (an HTTP range request),
  which is far more text than a CPU run can consume.

Each of the W workers gets its own contiguous shard of the training tokens and draws its own
random windows from it, with its own generator, so worker w never sees worker v's data. Validation
uses a fixed set of windows from the held-out split.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).resolve().parent.parent / "data"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TS_BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
TS_TRAIN_BYTES = 200 * 2**20
TS_VOCAB = 4096
EOT = "<|endoftext|>"


def _download(url: str, dest: Path, max_bytes: int | None = None) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "DeMo_Fixed_Band_Experiment"})
    if max_bytes:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest


def _pad64(n: int) -> int:
    return -(-n // 64) * 64


def load_shakespeare():
    text = _download(SHAKESPEARE_URL, DATA / "shakespeare" / "input.txt").read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], _pad64(len(chars)), {"tokeniser": "characters", "vocab": len(chars)}


def load_tinystories():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

    root = DATA / "tinystories"
    tok_path = root / f"bpe{TS_VOCAB}.json"
    cache = {s: root / f"{s}_bpe{TS_VOCAB}.npy" for s in ("train", "val")}
    if not all(p.exists() for p in cache.values()):
        train_txt = _download(TS_BASE + "TinyStoriesV2-GPT4-train.txt", root / "train_head.txt", TS_TRAIN_BYTES)
        val_txt = _download(TS_BASE + "TinyStoriesV2-GPT4-valid.txt", root / "valid.txt")
        raw = train_txt.read_text(errors="ignore")
        raw = raw[: raw.rfind(EOT)]                       # drop the story the range request cut
        if not tok_path.exists():
            tok = Tokenizer(models.BPE())
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tok.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(vocab_size=TS_VOCAB, special_tokens=[EOT],
                                          initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
            sample = raw[: 20 * 2**20].split(EOT)
            tok.train_from_iterator(sample, trainer)
            tok.save(str(tok_path))
        tok = Tokenizer.from_file(str(tok_path))
        eot = tok.token_to_id(EOT)
        for split, text in (("train", raw), ("val", val_txt.read_text(errors="ignore"))):
            stories = [s.strip() for s in text.split(EOT) if s.strip()]
            ids = []
            for i in range(0, len(stories), 10000):
                for enc in tok.encode_batch(stories[i:i + 10000]):
                    ids.extend(enc.ids)
                    ids.append(eot)
            np.save(cache[split], np.array(ids, dtype=np.uint16))
    tr, va = np.load(cache["train"]), np.load(cache["val"])
    return tr, va, _pad64(TS_VOCAB), {"tokeniser": f"byte-level BPE, {TS_VOCAB} tokens", "vocab": TS_VOCAB}


def load_synthetic():
    """Random tokens from a fixed seed: no download, for tests and CI smoke runs."""
    g = np.random.default_rng(0)
    return (g.integers(0, 60, 200_000).astype(np.uint16), g.integers(0, 60, 20_000).astype(np.uint16), 64,
            {"tokeniser": "synthetic", "vocab": 60})


def load(name: str):
    return {"shakespeare": load_shakespeare, "tinystories": load_tinystories, "synthetic": load_synthetic}[name]()


class Shards:
    """W disjoint contiguous shards of the training tokens, each with its own sampler."""

    def __init__(self, train: np.ndarray, workers: int, block: int, batch: int, seed: int):
        n = len(train) // workers
        self.shards = [torch.from_numpy(train[w * n:(w + 1) * n].astype(np.int64)) for w in range(workers)]
        self.block, self.batch = block, batch
        self.gens = [torch.Generator().manual_seed(seed * 1000 + w) for w in range(workers)]

    def batch_for(self, w: int):
        s = self.shards[w]
        i = torch.randint(len(s) - self.block - 1, (self.batch,), generator=self.gens[w])
        x = torch.stack([s[j:j + self.block] for j in i])
        y = torch.stack([s[j + 1:j + 1 + self.block] for j in i])
        return x, y

    def state(self):
        return [g.get_state() for g in self.gens]

    def set_state(self, states):
        for g, s in zip(self.gens, states):
            g.set_state(s)


def val_batches(val: np.ndarray, block: int, batch: int, n: int):
    """A fixed list of n validation batches (the same for every run)."""
    v = torch.from_numpy(val.astype(np.int64))
    g = torch.Generator().manual_seed(12345)
    out = []
    for _ in range(n):
        i = torch.randint(len(v) - block - 1, (batch,), generator=g)
        out.append((torch.stack([v[j:j + block] for j in i]), torch.stack([v[j + 1:j + 1 + block] for j in i])))
    return out


def describe(meta: dict) -> str:
    return json.dumps(meta)
