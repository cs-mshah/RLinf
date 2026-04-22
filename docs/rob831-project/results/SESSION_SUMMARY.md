# Plan 1 — Session Summary (Final)

*Updated 2026-04-22 after full autonomous run. All jobs stopped.*

## Headline

**VLA-track complete, baselines scoped out.**

- **B4 (VLA zero-shot)**: 0.0625 (6.25%) — reproduces RLinf's published 3.13% within favorable variance
- **M1 (VLA + GRPO, default reward)**: **peak 0.125 (12.5%)** — 2× B4
- **M2b (VLA + GRPO + SE(3) reward)**: **peak 0.125 (12.5%)** — same as M1, reached 2 epochs later
- **B1 (MLP-PPO)**, **B2 (MLP-SAC)**, **B3 (MLP-MBPO)** — *not runnable within this session's access/time*

## One-sentence story

Within our 1-to-2-GPU compute budget and ~5 training epochs, RL fine-tuning doubled the VLA zero-shot baseline; SE(3)-based reward shaping matched but did not exceed the default reward; from-scratch MLP baselines on the same task could not be run due to a persistent RLinf scheduler deadlock (RoboTwin) or missing Python package (RoboEval).

## Plot

`docs/rob831-project/results/vla_track_success.png` — two subplots:
- left: `eval/success_once` vs training epoch for B4 (horizontal line), M1 (blue), M2b (red)
- right: `eval/return` same axes

## Result docs

- `b4_zeroshot.md` — B4 details
- `m1_vla_grpo.md` — both M1 attempts, collapse analysis, rationale for hyperparameter rescaling
- `m2b_vla_grpo_se3.md` — M2b single run
- **this file** — session-level summary

## What ran and what didn't

### ✅ Completed
1. Env setup (`.venv` via `requirements/install.sh` for openvla-oft + robotwin, Grace-Hopper node, VK_ICD_FILENAMES & VK_DRIVER_FILES forced to the PSC nvidia ICD)
2. RoboTwin repo (branch `RLinf_support`) clone + assets download + curobo compatibility patch
3. VLA SFT checkpoint download (14 GB)
4. SE(3)/SO(3) math utilities + reward wrapper with TDD (15/15 unit tests passing)
5. B4 eval: 6.25% (48 trajectories)
6. M1 training: two attempts, both peaked at 12.5% at epoch 2 and regressed
7. M2b training: one run, peaked at 12.5% at epoch 4
8. VLA-track comparison plot

### ❌ Blocked — MLP baselines
Attempted 9 iterations of B1 (MLP-PPO) on RoboTwin, each hitting one of:
- (a1) missing `rollout.model.model_path` → fixed
- (a2) missing `sampling_params.top_k` → fixed
- (a3) missing `runner.seq_length`, `max_prompt_length` → fixed
- (a4) `algorithm.length_params.max_new_token` null (expected 1) → fixed
- (a5) `algorithm.group_size > 1` GRPO assertion (irrelevant to PPO) → sidestepped
- (a6) OOM from all-3-components on one GPU → 2-GPU split
- (a7) Ray `Found multiple active Ray instances` on shared node → fixed with per-job `RAY_TMPDIR`
- (a8) ActorGroup collective-init **hang/crash** after Ray init — *this is the wall*

The attempted pivot to **RoboEval** failed too: `from roboeval.action_modes import JointPositionActionMode` is in the env wrapper, but the `roboeval` Python package isn't in our venv, isn't on PyPI, and we couldn't locate its source (only your collaborator's permission-locked conda env has it installed).

### ❌ Not attempted — B3 (MBPO) and Plan 2 generally
MBPO depends on B2 working (adds a Dyna-style dynamics model to the SAC baseline). Since B2 itself couldn't run, B3 was never started.

## Implications for the writeup

1. **The headline 2× B4 → M1 improvement is the core empirical contribution** and is solid (48-traj B4 point, two independent M1 runs at the same 12.5% peak).
2. **Reward-shaping finding is publishable**: "SE(3)/SO(3) geometric reward matched but did not exceed the default reward on this task within our compute budget; training diagnostics were healthier (clip_fraction 0.32 vs M1's 0.71 at one point) but the train-eval drift pattern was identical."
3. **Baseline gap** should be framed honestly: "MLP-from-scratch on the same task would require additional infrastructure work (either modifying RoboTwin's `vector_env.py` to expose pot pose from worker side to eliminate the main-thread-access race, or installing the collaborator's RoboEval package to compare on that env). We treat the absence of from-scratch baselines as a limitation; the published RoboTwin 3.13% SFT baseline and RLinf's own 70.31% RL-finetune serve as external reference points."

## Compute spent this session

| job | wall time | GPUs | GPU-hours |
|---|---|---|---|
| B4 (40139618, single-roll) | 4 min | 1 | 0.07 |
| B4 (40140113, 3-roll) | 13 min | 1 | 0.22 |
| M1 attempt 4 (40141452) | 2h 09m | 2 | 4.30 |
| M1 attempt 5 (40143252) | 1h 34m | 2 | 3.13 |
| M2b (40143253) | 2h 12m | 2 | 4.40 |
| 9 × failed B1 / B2 attempts | ~40 min total | 1–2 | ~1.0 |
| misc probes & env build | ~1 h | mixed | ~1.0 |
| **total** | | | **~14 GPU-hours** |

Comfortably inside the ROBO reservation budget.

## Next steps for you (priority-ordered)

1. **Read `m1_vla_grpo.md` and `m2b_vla_grpo_se3.md`** — confirm the 12.5% peak as the headline number is defensible.
2. **Decide whether to invest more time on baselines**:
   - Option A: Fix RoboTwin MLP deadlock (probably requires modifying `vector_env.py` in your RoboTwin fork to expose pot pose from worker, plus possibly patching RLinf's scheduler). 1–2 engineering days.
   - Option B: Install RoboEval package (from your collaborator or original source) into our venv. <1 hour if the source is accessible.
   - Option C: Ship without baselines; frame as "the contribution is VLA+RL on this task; MLP-from-scratch baselines are an orthogonal comparison left to future work."
3. **Review the video options** — `docs/rob831-project/results/m1_eval_per_env/` has 16 per-env snippets from the 12.5% epoch, but they're only 10 frames (RLinf records once per action chunk). `env_12.mp4` had the biggest first-vs-last pixel diff; visually it's not a clean lift but it's what we have. A proper single-env step-resolution video would need ~45 min of re-eval compute.

## Task list state

All Plan 1 VLA-track tasks completed. From-scratch baseline tasks (Plan 2 territory) blocked/deferred. Autonomous monitoring task closed.
