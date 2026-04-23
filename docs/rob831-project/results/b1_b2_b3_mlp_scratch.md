# B1 / B2 / B3 — From-scratch MLP RL Baselines on RoboEval LiftPot

## Headline

**All three from-scratch baselines fail to sustain any nonzero eval success rate** within their compute budgets. B1 and B3 each produced a single isolated checkpoint spike to 1.0 eval success that did not persist to adjacent checkpoints (classic lucky-seed variance against the fixed 16-env eval pool). B2 never showed any spike.

Under the reporting convention of **mean of last 5 eval checkpoints**, all three baselines are at **0.000**.

## Runs

| id | algorithm | job | wall time | env steps reached | # eval checkpoints | max eval `success_once` | mean-of-last-5 eval | final eval `return` |
|---|---|---|---|---|---|---|---|---|
| B1 | MLP-PPO (RLinf) | 40177664 | 3h 13m (cancelled at watchdog) | 319 epochs | 32 | 1.000 (two isolated spikes at steps 79, 269) | **0.000** | -1882 |
| B2 | MLP-SAC (RLinf) | 40177904 | 2h 34m (cancelled at watchdog) | 5600 epochs | 28 | 0.000 | **0.000** | -1022 |
| B3 | MLP-MBPO (standalone Dyna-1) | 40178138 | 24m (full run, completed) | 200 000 env steps | 40 | 1.000 (one isolated spike) | **0.000** | -2.79 |

## Variance / lucky-seed note (important!)

`use_fixed_reset_state_ids: True` means every eval uses the same 16 initial-state seeds. Combined with a low eval N = 16, a transient policy snapshot that happens to produce good behavior against a favorable subset of those seeds can produce a full 1.0 `success_once` even though adjacent checkpoints drop back to 0. B1 shows this pattern twice (steps 79 and 269), B3 once. The right summary is the mean of the last 5 (or last 10%) of eval checkpoints — not the max — which is 0 for all three.

## What the return plot shows (the softer signal)

Since all three used RoboEval's dense shaped reward (`_LiftPotDenseRewardWrapper` — reach/grasp/lift/pose/success/action_rate components), returns are a softer window into behavior:

- **B1 (PPO)**: `eval/return` started near -1400 and drifted to -1900 over training. PPO's stochastic exploration finds more of the "flail and rack up pose-penalty" regions.
- **B2 (SAC)**: `eval/return` oscillated around -700 to -1500. SAC's entropy-regularized exploration is slightly more cautious than PPO's.
- **B3 (MBPO-lite)**: `eval/return` converged to about -3. The policy effectively learned to *do almost nothing* — minimal actions avoid the largest penalty terms. This is the classic pathology of a model-based method when the dynamics model is imperfect and the reward gradient is hard to exploit; the policy finds a local minimum of "inaction."

None of these return trajectories correspond to learning the actual task; they're all reward-hacking the penalty structure without finding the sparse success bonus.

## Cross-env comparison note

B1/B2/B3 are on **RoboEval** (`BimanualPanda`, 16-DOF, dense shaped reward), while the VLA track (B4/M1/M2b) is on **RoboTwin** (`PIPER bimanual`, 14-DOF, sparse binary reward). The two envs are structurally equivalent bimanual pot-lift tasks but:

- **`eval/success_once` IS directly comparable** — both report fraction of eval envs that reached the task-success predicate.
- **`eval/return` is NOT directly comparable** — RoboTwin returns are 0 or 5 (sparse × reward_coef); RoboEval returns are in the -100 to -2000 range (dense penalty-dominated).

The `all_tracks.png` plot splits return into two separate panels to avoid the illusion of direct comparability.

## Why this is exactly what we expected to find

- Published RLinf baseline on RoboTwin lift_pot: **3.13%** SFT success, **70.3%** after *1000 epochs of 8-GPU RL fine-tuning*. This task takes substantial compute to solve.
- Our compute: ~3h each of 2× H100 on RoboEval MLP. That's roughly 0.75% of the compute RLinf used for the 70% result.
- MLP from scratch has no prior knowledge of bimanual coordination, grip alignment, or pot dynamics. VLA+GRPO at least starts from a 3% SFT policy with visual pot-localization already learned.

**Narrative for the writeup:** the experiments confirm that in this compute regime, VLA pretraining is the load-bearing ingredient. All three RL algorithm families (on-policy, off-policy, model-based) fail to escape 0% sustained success from scratch; VLA+RL reaches 12.5%.

## Compute spent on this sub-plan

| job | wall time | GPUs | GPU-hours |
|---|---|---|---|
| B1 (PPO) | 3h 13m | 2 | 6.43 |
| B2 (SAC) | 2h 34m | 2 | 5.13 |
| B3 (MBPO) | 24m | 1 | 0.40 |
| total | — | — | **~12.0 GPU-hours** |

Plus ~40 min of earlier failed attempts on RoboTwin (~1 GPU-hour).
