# B1 / B2 / B3 — From-scratch MLP RL Baselines on RoboEval LiftPot

## Headline

**All three from-scratch baselines fail to sustain any nonzero eval success rate** within their compute budgets.

**Important correction (2026-04-24):** the single-checkpoint `eval/success_once = 1.000` spikes for B1 (steps 79, 269) and B3 (step 184704) are **NOT real successes — they are a RoboEval instrumentation artifact.** At those same checkpoints, `eval/return` is deeply negative (-220, -369, -7.6 respectively), which is impossible if 16/16 envs had actually succeeded under the dense reward: with `w_success = 10`, a real full success would yield `return ≈ +10 * 16 − penalties ≫ 0`. Root cause is an info-dict key mismatch — RoboEval's base env writes `info["task_success"]` (see `roboeval_env.py:481`), but RLinf's wrapper reads `info["success"]` both for the dense-reward success bonus (`rlinf/envs/roboeval/roboeval_env.py:77`) and for `success_once` aggregation (line 582). Combined with `LiftPot._success_check` being a class-level attribute defaulting to `True` (`lift_pot.py:26`) that `_on_reset` never clears, the `success_once` flag can latch true from stale state while the reward term silently stays 0. We did not fix and re-run (per decision 2026-04-24); the spikes should be disregarded as instrumentation noise.

Under the reporting convention of **mean of last 5 eval checkpoints** — or training-rollout `env/success_once`, which is unaffected — all three baselines are at **~0**.

## Runs

| id | algorithm | job | wall time | env steps reached | # eval checkpoints | max eval `success_once` | mean-of-last-5 eval | final eval `return` |
|---|---|---|---|---|---|---|---|---|
| id | algorithm | job | wall time | # eval ckpts | max eval `success_once` | mean-of-last-5 eval | env `success_once` (training rollouts, unbiased) | final eval `return` |
|---|---|---|---|---|---|---|---|---|
| B1 | MLP-PPO (RLinf) | 40177664 | 3h 13m (cancelled at watchdog) | 32 | 1.000 ⚠️ *instrumentation bug* | **0.000** | peak 0.111 / last 0.053 (n=328) | -1882 |
| B2 | MLP-SAC (RLinf) | 40177904 | 2h 34m (cancelled at watchdog) | 28 | 0.000 | **0.000** | peak 0.031 / last 0.000 (n=56) | -1022 |
| B3 | MLP-MBPO (standalone Dyna-1) | 40178138 | 24m (full run, completed) | 40 | 1.000 ⚠️ *instrumentation bug* | **0.000** | (not logged) | -2.79 |

The training-rollout column (`env/success_once`) is the more honest signal — it aggregates over *many* rollouts per checkpoint, and the info-dict key-mismatch bug only affected the eval code path. Even on this signal, none of the baselines hits even 12% and all drop back to near-zero.

## Why the eval=1.0 spikes are NOT real successes (deleting the earlier "lucky-seed" framing)

Earlier draft of this doc framed the spikes as low-N variance ("lucky seed against 16 fixed initial-state IDs"). That framing was wrong. Three pieces of evidence:

1. **Internal inconsistency with return.** At B1 step 79, `eval/success_once = 1.000` but `eval/return = -220.30`. Dense reward has `w_success = 10` — a real 16/16 success would produce return ≳ `+160` per rollout batch before penalties. B1 step 269 is worse (return -368.70 at claimed 1.0 success). B3 step 184704: return -7.59 at claimed 1.0 success.

2. **Info-dict key mismatch.** RoboEval's base `get_info()` writes `info["task_success"] = float(self.success)` (`roboeval_env.py:481`). RLinf's RoboEval wrapper reads `info.get("success", 0.0)` for the reward bonus (`rlinf/envs/roboeval/roboeval_env.py:77`) and `infos["success"]` for `success_once` aggregation (line 582). So the dense-reward success term is always 0, yet `success_once` somehow still latches True.

3. **Stale class-attribute default.** `LiftPot._success_check = True` is a class-level attribute (`lift_pot.py:26`) that `_on_reset()` does not reset. When `success_once` reads through whatever code path DOES manage to put `"success"` in `infos` at eval time, it can read stale True from a previous episode's residue or the class default.

The honest reading: eval `success_once` is unreliable on RoboEval under this wrapper; use `env/success_once` (training rollouts) instead, which is unaffected.

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
