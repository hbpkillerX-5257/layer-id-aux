# layer-id-aux

Does a transformer get better if its hidden states are trained to reveal which layer they came from?

The auxiliary term is a shared linear classifier. It reads the residual stream after each block and predicts the layer index. That cross-entropy is added to the usual next-character loss:

```
L = L_ce + λ * L_layer
```

A cheap control trains the same classifier on detached hidden states, so the probe accuracy is measured and the trunk does not see that gradient.

## Modes

| mode | trunk loss | what the probe does |
| --- | --- | --- |
| `baseline` | next-character cross-entropy | trained on detached states |
| `layer_id` | cross-entropy plus `λ` times layer-index cross-entropy | same readout, gradient enters the trunk |
| `cosine` | cross-entropy plus `λ` times mean cosine of consecutive layers | trained on detached states |
| `brier_head` | cross-entropy plus `λ` times Brier score of a scalar confidence head | head predicts whether the argmax token is correct |
| `brier_logit` | cross-entropy plus `λ` times Brier score of the softmax top probability | the probability being scored is the model's own top guess |
| `brier_full` | cross-entropy plus `λ` times the multiclass Brier score of the full softmax | no extra head |

`depth_energy` is the fraction of between-layer mean variance that sits in the top singular direction. A value near 1 means the layers separated by a single depth stamp.

## Run

```bash
python -m src.train --mode baseline --seed 1
python -m src.train --mode layer_id --lam 0.05 --seed 1
python -m src.train --mode cosine --lam 0.1 --seed 1
```

The default run is a character-level TinyShakespeare model: 8 layers, width 256, 4 heads, 4000 steps. It downloads the corpus into `data/` on first use. Metrics land in `results/<run>/metrics.jsonl` and `summary.json`.

Kaggle's Python image already has PyTorch. Locally:

```bash
pip install -r requirements.txt
```

## Kaggle

`scripts/launch_kaggle.py` reads access tokens from `~/.kaggle/access_token` and `~/.kaggle/access_token_1` … `access_token_6`. It identifies each token, keeps one token per username, and pushes one private GPU kernel per account. The kernel clones this public repo at the pinned commit and trains there.

```bash
python scripts/launch_kaggle.py --quota-only
python scripts/launch_kaggle.py
```

The scheduled matrix is two baseline seeds, two `layer_id` seeds at λ = 0.05, one `layer_id` seed at λ = 0.2, and one cosine penalty at λ = 0.1.
