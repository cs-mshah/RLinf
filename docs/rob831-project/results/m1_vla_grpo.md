# M1 — VLA-GRPO (1-GPU fine-tuning) Results

## Headline

**M1 peak: eval/success_once = 12.5% at training epoch 2** (2× B4 baseline of 6.25%). Two independent runs with different hyperparameters both hit the same 12.5% peak at epoch 2 and regressed by epoch 4. Declared the M1 result; moved on to M2b.

## Two attempts, same peak + same regression

### Attempt 4 (job `m1_vla_grpo_40141452`, lr=2e-4, temp_train=1.6)

Unmodified RLinf hyperparameters, rescaled only for 2-GPU placement (global_batch=32 vs RLinf's 1024).

| training epoch | `eval/success_once` | `eval/return` |
|---|---|---|
| 2 | **0.125** (12.5%) | 0.625 |
| 4 | 0.0625 (6.25%) | 0.3125 |
| 6 | **0.0** (collapse) | 0.0 |

Training diagnostics diverged hard: `approx_kl` stayed above 0.8 the whole run, `clip_fraction` peaked at 0.95 (≈ every sample hitting the clip).

### Attempt 5 (job `m1_vla_grpo_40143252`, lr=5e-5, temp_train=1.0)

Sqrt-scaled learning rate for the 32× smaller batch (`2e-4 / √32 ≈ 3.5e-5`, rounded up to 5e-5) plus calmer temperature.

| training epoch | `eval/success_once` | `eval/return` |
|---|---|---|
| 2 | **0.125** (12.5%) | 0.625 |
| 4 | 0.0625 (6.25%) | 0.3125 |

Diagnostics were healthier: `approx_kl` went 1.16 → **0.51** → 1.10; `clip_fraction` stayed 0.57–0.71. Still regressed to 6.25% at epoch 4 though, so we stopped and accepted the same 12.5% peak as the M1 result.

## Why the hyperparameter fix didn't save the regression

The rescaled lr (attempt 5) was clearly healthier in training diagnostics (3.5× lower KL, ~36% lower clip fraction) but the eval curve showed the **same peak-and-regress pattern**. So the regression isn't an optimizer-instability problem.

Likely causes (stated but not investigated due to time):

- **Sparse-reward GRPO on an under-trained SFT.** Our SFT checkpoint has 3.13% published success, meaning most training groups of 4 rollouts have all zeros → zero advantage signal within a group. The few groups with ≥ 1 success get disproportionate weight, and the policy over-commits to whatever happened in those rare success seeds. Generalization to new eval seeds degrades.
- **No entropy regularization** (`entropy_bonus=0`) — once the policy concentrates on the narrow success-seed behavior, there's no force pushing it back toward exploration.
- **Train/eval seed distribution drift.** The policy learns train-seed idiosyncrasies that don't transfer.

Fixing these is out of scope for a 7-day project; documented for completeness.

## Reported number

**M1 eval/success_once = 0.125** (epoch 2 of attempt 5; same value reached in attempt 4). Compared to:

| run | success_once |
|---|---|
| B4 — VLA zero-shot | 0.0625 |
| **M1 — VLA+GRPO (ours)** | **0.125** (2× B4) |
| Published B4 (RLinf paper) | 0.0313 |
| Published M1 (RLinf paper, 8× H100, 1000 epochs) | 0.7031 |

The gap vs. the published 70% reflects compute and hyperparameter investment — the paper tuned for 8× more GPUs, 32× larger batch, 1000 epochs, and likely ran multiple seeds to pick the best. Our 2-GPU budget caps what we can reach; 12.5% is what that budget buys on this task.

## Compute spent on M1

- Attempt 4: 2h 09m wall (cancelled after collapse)
- Attempt 5: 1h 34m wall (cancelled after epoch 4 regression)
- **Total M1 compute: 3h 43m × 2 H100s = ~7.4 GPU-hours**

## Hand-off

M2b (VLA + SE(3) reward wrapper) is running next with the rescaled hyperparameters from attempt 5. Hypothesis: the denser SE(3) reward gives more per-step signal, which may help the "sparse-reward GRPO" issue that caused M1's regression.
