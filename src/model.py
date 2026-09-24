"""Small decoder-only transformer with a shared layer-index head.

The head reads a hidden state and predicts which block produced it. Callers
decide whether that loss is allowed to update the trunk.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_head ({n_head})")
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = self.attn_drop(torch.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model)
        self.proj = nn.Linear(4 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class LayerIdHead(nn.Module):
    """Shared linear classifier from a hidden state to a layer index."""

    def __init__(self, d_model: int, n_layer: int):
        super().__init__()
        self.proj = nn.Linear(d_model, n_layer)
        self.n_layer = n_layer

    def forward(self, hiddens: list[torch.Tensor], stopgrad: bool) -> tuple[torch.Tensor, torch.Tensor]:
        losses = []
        correct = []
        total = []
        for index, hidden in enumerate(hiddens):
            features = hidden.detach() if stopgrad else hidden
            logits = self.proj(features)
            target = torch.full(features.shape[:2], index, device=features.device, dtype=torch.long)
            losses.append(F.cross_entropy(logits.view(-1, self.n_layer), target.view(-1)))
            correct.append((logits.argmax(-1) == index).sum())
            total.append(torch.tensor(target.numel(), device=features.device))
        loss = torch.stack(losses).mean()
        accuracy = torch.stack(correct).sum() / torch.stack(total).sum().clamp(min=1)
        return loss, accuracy


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        d_model: int,
        n_head: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block_size = block_size
        self.n_layer = n_layer
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.layer_id_head = LayerIdHead(d_model, n_layer)
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.proj.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        b, t = idx.shape
        if t > self.block_size:
            raise ValueError(f"sequence length {t} exceeds block size {self.block_size}")
        positions = torch.arange(t, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions)[None, :, :])
        hiddens: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            hiddens.append(x)
        logits = self.lm_head(self.ln_f(x))
        return logits, hiddens


def task_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def consecutive_cosine(hiddens: list[torch.Tensor]) -> torch.Tensor:
    sims = []
    for left, right in zip(hiddens, hiddens[1:]):
        left_n = F.normalize(left, dim=-1, eps=1e-6)
        right_n = F.normalize(right, dim=-1, eps=1e-6)
        sims.append((left_n * right_n).sum(dim=-1).mean())
    if not sims:
        return torch.zeros((), device=hiddens[0].device)
    return torch.stack(sims).mean()


@torch.no_grad()
def depth_direction_energy(hiddens: list[torch.Tensor]) -> float:
    """Share of between-layer mean variance sitting in the top singular direction.

    Near 1 means a single depth stamp separates the layers. Near 1/n_layer means
    the layer means are spread across many directions.
    """
    means = torch.stack([hidden.float().mean(dim=(0, 1)) for hidden in hiddens])
    means = means - means.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(means)
    energy = singular.square()
    return (energy[0] / energy.sum().clamp(min=1e-8)).item()
