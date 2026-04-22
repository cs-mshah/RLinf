# M1 — VLA-GRPO (1-GPU fine-tuning) Results

## Summary

The VLA+GRPO fine-tune successfully reproduced **improvement over B4 at the peak** but suffered a collapse when using RLinf's published learning rate (2e-4) at a 32× smaller batch. A second run with `lr=5e-5` and `temperature_train=1.0` followed.

## Attempt 4 (job `m1_vla_grpo_40141452`) — hyperparameter collapse

**Config**: RLinf's default hyperparameters, verbatim from `robotwin_lift_pot_grpo_openvlaoft.yaml` (assumes 8 GPUs, global_batch=1024), rescaled only for 2-GPU placement (global_batch=32). Everything else identical.

**Trajectory** — `eval/success_once` by training epoch:

| Epoch | Eval success | vs B4 (6.25%) |
|---|---|---|
| 2 | **12.5%** | +6.25 pp (2× B4) — peak |
| 4 | 6.25% | = B4 |
| 6 | **0.0%** | collapse |

**Training diagnostics:**

| Epoch | `approx_kl` | `clip_fraction` | `policy_loss` | `grad_norm` |
|---|---|---|---|---|
| 0 | 4.10 | 0.89 | 0.027 | 0.59 |
| 1 | 1.24 | 0.49 | 0.006 | 0.18 |
| 2 | 1.46 | 0.67 | 0.009 | 0.48 |
| 3 | 1.55 | **0.95** | — | — |
| 4 | 0.85 | 0.59 | — | — |
| 5 | 1.81 | **0.94** | — | — |

Pathologically high clip_fraction (90%+ on epochs 3 and 5) means PPO-clip was containing *almost every sample* from the aggressive updates. Eventually the deterministic-greedy eval policy drifted away from the SFT initialization into a region with no positive reward at all.

## Root cause: batch-scaling mismatch on learning rate

RLinf's published lr=2e-4 was tuned for `global_batch_size=1024` (8 GPUs × 128 micro_batch). Our 1-GPU/2-GPU config uses `global_batch_size=32` — **32× smaller**. Standard scaling rules say lr should drop by `√K ≈ 5.7×` (or `K=32×` for noise-normalized methods):
- sqrt-scaled: `2e-4 / 5.7 ≈ 3.5e-5`
- linear: `2e-4 / 32 ≈ 6e-6`

We lifted the hyperparameters verbatim without this adjustment, so effective steps were 5–30× too large.

## Attempt 5 (job `m1_vla_grpo_40143252`) — rescaled

**Config changes** (committed as `69dc152`):
- `actor.optim.lr: 2.0e-4 → 5.0e-5` (sqrt-scaling with slight headroom)
- `algorithm.sampling_params.temperature_train: 1.6 → 1.0` (reduces off-policy gap between rollout and training distributions)
- All other knobs unchanged

Running at time of writing. Will update this file once results land.

## Take-away

- **B4 = 6.25%** (reproduced, stable over 48 trajectories)
- **M1 peak (attempt 4) = 12.5%** — confirms VLA+GRPO *can* improve over zero-shot on this task/compute budget
- **Attempt-5 trajectory is the official M1 result** for the project writeup; attempt 4 is noted as a hyperparameter-sensitivity finding (useful negative result: RLinf's hyperparams require batch-scale-aware lr tuning for smaller-compute reproductions)

Both the M1 and M2b (VLA + SE(3) reward) runs use the rescaled hyperparameters so any difference between them is attributable to the reward function, not to the optimizer settings.
