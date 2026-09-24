"""Character-level TinyShakespeare, cached under data/."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text(data_dir: Path) -> str:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "input.txt"
    if not path.exists() or path.stat().st_size == 0:
        urllib.request.urlretrieve(URL, path)
    return path.read_text(encoding="utf-8")


class CharData:
    def __init__(self, text: str, block_size: int, val_frac: float = 0.1):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.vocab_size = len(self.stoi)
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n_val = max(block_size + 2, int(len(ids) * val_frac))
        if n_val >= len(ids) - block_size - 2:
            raise ValueError("corpus is too small for the requested block size")
        self.train = ids[:-n_val]
        self.val = ids[-n_val:]
        self.block_size = block_size

    def batch(self, split: str, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.val
        hi = len(data) - self.block_size - 1
        ix = torch.randint(0, hi, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)
