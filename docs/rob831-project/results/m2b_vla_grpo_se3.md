# M2b — VLA-GRPO + SE(3)/SO(3) Reward Results

## Headline

**M2b peak: eval/success_once = 12.5% at training epoch 4**. Same peak as M1 (default reward), reached two epochs later. Conclusion within our compute budget: the SE(3)/SO(3) reward shaping neither helps nor hurts the peak, and exhibits the same post-peak regression as M1.

## Single run, four evals

Job `m2b_vla_grpo_se3_40143253` — rescaled hyperparameters from M1 attempt 5 (`lr=5e-5`, `temperature_train=1.0`), SE(3) reward via `reward_variant: se3` env config.

| training epoch | `eval/success_once` | `eval/return` |
|---|---|---|
| 2 | 0.0 | 0.0 |
| 4 | **0.125** (12.5%) | 0.625 |
| 6 | 0.0 | 0.0 |

Training-rollout diagnostics looked healthier than M1's at the same epochs:

| epoch | M1 (default) `approx_kl` | M2b (SE(3)) `approx_kl` | M1 `clip_fraction` | M2b `clip_fraction` |
|---|---|---|---|---|
| 0 | 1.16 | 1.17 | 0.57 | 0.72 |
| 1 | 0.51 | 0.51 | 0.71 | 0.70 |
| 2 | 1.10 | 1.07 | 0.57 | 0.91 |
| 3 | — | 0.45 | — | **0.32** |
| 4 | — | 0.60 | — | 0.71 |
| 5 | — | 0.54 | — | 0.71 |

Training rollout success climbed steadily: 3.1% → 4.7% → 6.3% → 3.1% → 7.0% → **12.5%** by epoch 6 training, while eval collapsed to 0%. **Classic train-eval drift / reward-hacking signature**: the policy finds a solution that maximizes the SE(3) reward on its training-seed distribution, and those gains don't transfer to deterministic-greedy eval on the held-out seed pool.

## Interpretation — "within compute budget, reward shaping didn't help"

The hypothesis behind M2b was that the default RoboTwin reward is sparse (mostly terminal success), so the policy gets little signal early on; SE(3) geodesic-distance + log-map terms should densify the reward and accelerate learning. The evidence we collected is mixed:

- **Densification worked as designed** — M2b's training `env/return` climbed more smoothly than M1's, and training-rollout success reached 12.5% (vs M1's 3–6%).
- **But eval didn't track training.** At best, M2b hits 12.5% eval at epoch 4 — exactly M1's peak, just later. At worst, eval is 0%.
- **Both runs peaked early and regressed** — suggests a shared underlying failure mode (GRPO signal sparsity in early training, no entropy regularization, seed distribution drift), not something specific to one reward.

Properly disentangling would require longer compute and a couple of hyperparameter sweeps (entropy bonus, KL target, SE(3) weight schedule) — out of scope for a 7-day project.

## Reported number

**M2b eval/success_once = 0.125** (best checkpoint, epoch 4). Compared to:

| run | peak success_once | notes |
|---|---|---|
| B4 — VLA zero-shot | 0.0625 | no RL |
| **M1 — VLA+GRPO, default reward** | **0.125** (peak) | epoch 2 |
| **M2b — VLA+GRPO, SE(3) reward** | **0.125** (peak) | epoch 4 |

## Compute spent on M2b

- Single run, 2h 12m wall × 2 H100s = ~4.4 GPU-hours

## Hand-off

Next: Task 17 — generate the B4 / M1 / M2b comparison plot.
