"""Train a tiny Shakespeare LM with an auxiliary loss.

Modes
  baseline     Next-character cross-entropy. Probes are trained on detached
               features and do not update the trunk.
  layer_id     Add a layer-index classifier on the residual stream.
  cosine       Penalize cosine similarity between consecutive layers.
  brier_head   A scalar head predicts P(argmax is correct), scored with Brier,
               and the gradient enters the trunk.
  brier_logit  Brier score between the softmax's own top probability and
               whether that top guess was correct. Gradient enters the logits.
  brier_full   Multiclass Brier score of the whole softmax against the label.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from src.data import CharData, load_text
from src.model import (
    GPT,
    binary_brier,
    confidence_probability,
    consecutive_cosine,
    correctness,
    depth_direction_energy,
    multiclass_brier,
    softmax_confidence,
    task_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "baseline",
            "layer_id",
            "cosine",
            "brier_head",
            "brier_logit",
            "brier_full",
            "brier_layers",
            "brier_shift",
            "smooth",
        ),
        default="baseline",
    )
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


def expected_calibration_error(confidence: torch.Tensor, correct: torch.Tensor, n_bins: int = 10) -> float:
    edges = torch.linspace(0, 1, n_bins + 1)
    total = confidence.numel()
    error = confidence.new_zeros(())
    for index in range(n_bins):
        if index == 0:
            mask = confidence <= edges[1]
        else:
            mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        gap = (correct[mask].mean() - confidence[mask].mean()).abs()
        error = error + gap * (count / total)
    return error.item()


def safe_corr(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denom = left.std(unbiased=False) * right.std(unbiased=False)
    if float(denom) < 1e-8:
        return 0.0
    return float((left * right).mean() / denom)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(values[mask].mean())


@torch.no_grad()
def evaluate(model: GPT, data: CharData, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"val_ce": 0.0, "probe_acc": 0.0, "cosine": 0.0, "depth_energy": 0.0}
    top_probs = []
    head_probs = []
    outcomes = []
    future_heads = []
    future_outcomes = []
    for _ in range(args.eval_batches):
        x, y = data.batch("val", args.batch_size, device)
        logits, hiddens, features = model(x)
        _, accuracy = model.layer_id_head(hiddens, stopgrad=True)
        totals["val_ce"] += task_loss(logits, y).item()
        totals["probe_acc"] += accuracy.item()
        totals["cosine"] += consecutive_cosine(hiddens).item()
        totals["depth_energy"] += depth_direction_energy(hiddens)
        outcome_seq = correctness(logits, y).cpu()
        head_seq = confidence_probability(features, model.confidence_head, stopgrad=True).cpu()
        top_probs.append(softmax_confidence(logits).flatten().cpu())
        head_probs.append(head_seq.flatten())
        outcomes.append(outcome_seq.flatten())
        future_heads.append(head_seq[:, :-16].flatten())
        future_outcomes.append(outcome_seq[:, 16:].flatten())
    model.train()
    stats = {key: value / args.eval_batches for key, value in totals.items()}
    top = torch.cat(top_probs)
    head = torch.cat(head_probs)
    outcome = torch.cat(outcomes)
    right = outcome > 0.5
    stats.update(
        {
            "ece_softmax": expected_calibration_error(top, outcome),
            "ece_head": expected_calibration_error(head, outcome),
            "q_std": float(head.std(unbiased=False)),
            "pmax_std": float(top.std(unbiased=False)),
            "corr_q_pmax": safe_corr(head, top),
            "pmax_when_right": _masked_mean(top, right),
            "pmax_when_wrong": _masked_mean(top, ~right),
            "q_when_right": _masked_mean(head, right),
            "q_when_wrong": _masked_mean(head, ~right),
            "val_acc": float(outcome.mean()),
            "future_corr": safe_corr(torch.cat(future_heads), torch.cat(future_outcomes)),
            "future_ece": expected_calibration_error(torch.cat(future_heads), torch.cat(future_outcomes)),
        }
    )
    return stats


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
        logits, hiddens, features = model(x)
        ce = task_loss(logits, y)
        # Detached probes always train the readouts. Trunk gradients below are
        # the only path by which an auxiliary term can change the language model.
        aux, probe_acc = model.layer_id_head(hiddens, stopgrad=True)
        outcome = correctness(logits, y).detach()
        head_probability = confidence_probability(features, model.confidence_head, stopgrad=True)
        head_brier = binary_brier(head_probability, outcome)
        if args.mode == "cosine":
            cosine = consecutive_cosine(hiddens)
        else:
            with torch.no_grad():
                cosine = consecutive_cosine(hiddens)
        loss = ce
        if args.mode == "layer_id":
            aux_live, _ = model.layer_id_head(hiddens, stopgrad=False)
            loss = loss + args.lam * aux_live
        elif args.mode == "cosine":
            loss = loss + args.lam * cosine
        elif args.mode == "brier_head":
            live_probability = confidence_probability(features, model.confidence_head, stopgrad=False)
            loss = loss + args.lam * binary_brier(live_probability, outcome)
        elif args.mode == "brier_logit":
            loss = loss + args.lam * binary_brier(softmax_confidence(logits), outcome)
        elif args.mode == "brier_full":
            loss = loss + args.lam * multiclass_brier(logits, y)
        elif args.mode == "brier_layers":
            # Each layer's own next-token guess is scored with Brier. Early layers
            # can satisfy this by staying uncertain; they are not asked to match
            # the final prediction.
            layer_terms = []
            for hidden in hiddens:
                layer_logits = model.lm_head(model.ln_f(hidden))
                layer_outcome = correctness(layer_logits, y).detach()
                layer_terms.append(binary_brier(softmax_confidence(layer_logits), layer_outcome))
            loss = loss + args.lam * torch.stack(layer_terms).mean()
        elif args.mode == "brier_shift":
            # Features at t predict whether the guess at t+16 is correct, so the
            # head cannot read the margin of the token it is scoring.
            shift = 16
            if features.size(1) <= shift:
                raise SystemExit("brier_shift needs a block longer than 16")
            shifted = confidence_probability(features[:, :-shift], model.confidence_head, stopgrad=False)
            future = outcome[:, shift:]
            loss = loss + args.lam * binary_brier(shifted, future)
        elif args.mode == "smooth":
            # lam is the label-smoothing mass, kept as a control for brier_full.
            classes = logits.size(-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            one_hot = torch.nn.functional.one_hot(y, classes).to(log_probs.dtype)
            soft = (1.0 - args.lam) * one_hot + args.lam / classes
            loss = -(soft * log_probs).sum(dim=-1).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.grad is not None], 1.0)
        head_brier.backward()
        aux.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.steps - 1:
            stats = evaluate(model, data, args, device)
            stats.update(
                {
                    "step": step,
                    "lr": lr,
                    "train_ce": ce.item(),
                    "train_aux": aux.item(),
                    "train_brier": head_brier.item(),
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
                f"acc={stats['val_acc']:.3f} ece_p={stats['ece_softmax']:.3f} "
                f"ece_q={stats['ece_head']:.3f} corr={stats['corr_q_pmax']:.3f} "
                f"q_std={stats['q_std']:.3f}",
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
