"""Train a tiny Shakespeare LM with an optional layer-index auxiliary loss.

Modes
  baseline  Task cross-entropy only. A linear layer-index probe is trained on
            detached hidden states, so probe accuracy is logged and the trunk
            is unchanged by it.
  layer_id  Same probe, but its cross-entropy is added to the task loss and
            updates the trunk. This is the depth-stamp auxiliary loss.
  cosine    Penalize mean cosine similarity between consecutive layers. The
            detached probe is still trained, as a measurement only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from src.data import CharData, load_text
from src.model import GPT, consecutive_cosine, depth_direction_energy, task_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "layer_id", "cosine"), default="baseline")
    parser.add_argument("--lam", type=float, default=0.0, help="weight on the auxiliary term")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=400)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default="")
    return parser.parse_args()


def learning_rate(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def evaluate(model: GPT, data: CharData, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"val_ce": 0.0, "probe_acc": 0.0, "cosine": 0.0, "depth_energy": 0.0}
    for _ in range(args.eval_batches):
        x, y = data.batch("val", args.batch_size, device)
        logits, hiddens = model(x)
        aux, accuracy = model.layer_id_head(hiddens, stopgrad=True)
        totals["val_ce"] += task_loss(logits, y).item()
        totals["probe_acc"] += accuracy.item()
        totals["cosine"] += consecutive_cosine(hiddens).item()
        totals["depth_energy"] += depth_direction_energy(hiddens)
        del aux
    model.train()
    return {key: value / args.eval_batches for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    if args.mode == "baseline":
        args.lam = 0.0
    elif args.lam <= 0:
        raise SystemExit(f"{args.mode} requires --lam > 0")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    text = load_text(args.data_dir)
    data = CharData(text, args.block_size)
    model = GPT(
        vocab_size=data.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        d_model=args.d_model,
        n_head=args.n_head,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    run_name = args.run_name or f"{args.mode}-lam{args.lam:g}-s{args.seed}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "metrics.jsonl"
    n_params = sum(p.numel() for p in model.parameters())
    config = vars(args).copy()
    config["data_dir"] = str(args.data_dir)
    config["out_dir"] = str(args.out_dir)
    config["vocab_size"] = data.vocab_size
    config["n_params"] = n_params
    config["device"] = str(device)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(
        f"run={run_name} params={n_params/1e6:.2f}M vocab={data.vocab_size} "
        f"device={device} mode={args.mode} lam={args.lam} seed={args.seed}",
        flush=True,
    )

    model.train()
    best_val = float("inf")
    last_eval: dict[str, float] = {}
    t0 = time.time()
    for step in range(args.steps):
        lr = learning_rate(step, args.steps, args.warmup, args.lr)
        set_lr(optimizer, lr)
        x, y = data.batch("train", args.batch_size, device)
        logits, hiddens = model(x)
        ce = task_loss(logits, y)
        # Detached probe loss always trains the readout. The trunk only sees an
        # auxiliary gradient in layer_id mode, scaled by lam.
        aux, probe_acc = model.layer_id_head(hiddens, stopgrad=True)
        if args.mode == "cosine":
            cosine = consecutive_cosine(hiddens)
        else:
            with torch.no_grad():
                cosine = consecutive_cosine(hiddens)
        loss = ce + aux
        if args.mode == "layer_id":
            aux_live, _ = model.layer_id_head(hiddens, stopgrad=False)
            loss = loss + args.lam * aux_live
        elif args.mode == "cosine":
            loss = loss + args.lam * cosine

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.steps - 1:
            stats = evaluate(model, data, args, device)
            stats.update(
                {
                    "step": step,
                    "lr": lr,
                    "train_ce": ce.item(),
                    "train_aux": aux.item(),
                    "train_probe_acc": probe_acc.item(),
                    "train_cosine": cosine.item(),
                    "seconds": time.time() - t0,
                }
            )
            last_eval = stats
            best_val = min(best_val, stats["val_ce"])
            with log_path.open("a") as handle:
                handle.write(json.dumps(stats) + "\n")
            print(
                f"step={step:5d} train_ce={stats['train_ce']:.3f} val_ce={stats['val_ce']:.3f} "
                f"probe_acc={stats['probe_acc']:.3f} cosine={stats['cosine']:.3f} "
                f"depth_energy={stats['depth_energy']:.3f} aux={stats['train_aux']:.3f}",
                flush=True,
            )

    summary = {
        "run_name": run_name,
        "mode": args.mode,
        "lam": args.lam,
        "seed": args.seed,
        "best_val_ce": best_val,
        "final": last_eval,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
