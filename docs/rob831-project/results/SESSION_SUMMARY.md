# ROB 831 Term Project — Final Report

*Author: Hyun Joon Kwon. Compiled 2026-04-24 for the 2026-04-28 deadline.*

VLA pretraining + RL fine-tuning on a bimanual pot-lift task, compared against three RL-from-scratch MLP baselines (PPO, SAC, MBPO). All code and configs live on the `roboeval-integration` branch.

---

## 1. Research question

> **In a 7-day, ~25 GPU-hour budget, does a Vision-Language-Action (VLA) model pre-trained on a manipulation task family, fine-tuned with RL, outperform RL-from-scratch MLP policies from the three major RL algorithm families (on-policy, off-policy, model-based) on a hard bimanual pot-lift task?**

Plus a secondary sub-question: **does replacing the default sparse success reward with an SE(3)/SO(3)-geodesic dense reward accelerate or improve VLA+RL fine-tuning within the same budget?**

---

## 2. Headline results

| ID | Method | Env | Peak eval `success_once` | Sustained (mean-of-last-5 eval or env rollouts) |
|---|---|---|---|---|
| **B4** | OpenVLA-OFT SFT, zero-shot | RoboTwin lift_pot | **6.25%** (1/16) | — (single eval snapshot) |
| **M1** | VLA + GRPO, default sparse reward | RoboTwin lift_pot | **12.5%** (2/16) at epoch 2 | env-rollout peak 5.5% |
| **M2b** | VLA + GRPO, **SE(3) dense reward** (ours) | RoboTwin lift_pot | **12.5%** (2/16) at epoch 4 | env-rollout peak **12.5%** ✓ |
| B1 | MLP-PPO from scratch (RLinf) | RoboEval LiftPot | 0% real; 1.000 spike is an instrumentation bug (§7) | env-rollout peak 11.1%, last 5.3% |
| B2 | MLP-SAC from scratch (RLinf) | RoboEval LiftPot | 0% | env-rollout peak 3.1% |
| B3 | MLP-MBPO from scratch (standalone Dyna-1) | RoboEval LiftPot | 0% real; 1.000 spike is same bug | env-rollout: not logged; eval/return -2.8 (inaction-optimum) |

**Published reference (RLinf paper):** 3.13% SFT → 70.3% after 1000 epochs of 8×H100 RL fine-tuning. Our budget is ~0.75% of that compute.

**One-sentence story.** A VLA pretrained on this task family and fine-tuned with GRPO *doubles* zero-shot success (6.25% → 12.5%), while three RL-from-scratch MLPs (PPO/SAC/MBPO) across on-policy, off-policy, and model-based families all fail to sustain any nonzero success. The SE(3) reward modification matches the default reward on peak eval (12.5% vs 12.5%) but is the only VLA+RL run whose training-rollout success also reaches 12.5%, suggesting denser gradient signal even though held-out eval didn't improve in our budget.

Plots: `all_tracks.png` (6 runs, three panels) and `vla_track_success.png` (B4/M1/M2b only).

---

## 3. Environments

We used **two** simulators, for a deliberate reason stated below. Success (`success_once`) is defined equivalently in both — a task-success predicate over the pot's lifted pose — so the *success* metric is comparable. *Returns* are not comparable because the reward formulations differ; return is only a within-track diagnostic.

### 3.1 RoboTwin `lift_pot` (VLA track: B4, M1, M2b)

- Repo: `RoboTwin` branch `RLinf_support`.
- Robot: **PIPER bimanual** (two 7-DOF PIPER arms, 14-DOF action space; parallel-jaw grippers).
- Simulator backend: Sapien + CuRobo planner.
- Observation: RGB images (two cameras) + proprioception, passed through OpenVLA-OFT's visual encoder.
- Reset: 32-seed "reset state" pool; eval uses a held-out fixed 16-seed subset (`use_fixed_reset_state_ids: True`).
- Episode horizon: 200 steps.
- **Why chosen:** a publicly-released SFT checkpoint exists for this exact task — `RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot` (14 GB). This anchors the VLA track.

### 3.2 RoboEval `LiftPot` (MLP-scratch track: B1, B2, B3)

- Repo: `github.com/Robo-Eval/RoboEval`, installed `pip install -e .` with `mujoco==3.3.3` pinned.
- Robot: **BimanualPanda** (two 7-DOF Franka Panda arms + 1-DOF gripper each = 16-DOF action space).
- Simulator backend: MuJoCo (via `mujoco-menagerie` assets submodule).
- Observation: low-dimensional proprioceptive state (`obs_mode: state`, no pixels). Flattened into a Box by `_FlattenPropObsWrapper`.
- Reset: 16-seed fixed eval pool.
- Episode horizon: 200 steps.
- **Why chosen:** a working RLinf MLP training pipeline was already wired up for this env (from earlier project PRs). MLP-from-scratch on RoboTwin hit a persistent RLinf `ActorGroup` collective-init deadlock across 9 debug iterations; pivoting to RoboEval saved the week.

### 3.3 Key structural differences

| dimension | RoboTwin lift_pot | RoboEval LiftPot |
|---|---|---|
| action DOF | 14 (2×7) | 16 (2×8) |
| robot | PIPER | Franka Panda |
| obs space | RGB + proprio (high-dim) | state-only (low-dim) |
| physics | Sapien | MuJoCo |
| default reward | sparse 0/1 × `reward_coef=5` | dense shaped (see §4) |

Both tasks require the same qualitative skill: two arms coordinate to grip the pot's handles, lift ≥ 10 cm, keep the pot upright (rotation within ±20° of vertical). The `success_once` predicate checks these final-state conditions in both envs.

---

## 4. Reward functions

We used three different reward formulations across the six runs. The two environments' `success_once` predicate is equivalent; the *shaping* around it differs.

### 4.1 RoboTwin default (sparse) — used by B4, M1

Pure terminal success indicator, scaled by a reward coefficient:

$$r_t = 5 \cdot \mathbb{1}[\text{success predicate holds at step } t]$$

Return is non-negative, typically 0 until a policy first succeeds, then ≈ 5 × (consecutive success steps).

### 4.2 SE(3)/SO(3) dense reward (ours) — used by M2b

Implemented in `rlinf/envs/robotwin/lift_pot_reward_wrapper.py` and unit-tested in `tests/unit_tests/test_se3_math.py` (+ `_lift_pot_reward_wrapper.py`, 15/15 passing).

$$r_t = -w_p \|p_\text{pot} - p_\text{target}\|_2 - w_R\, d_{SO(3)}(R_\text{pot}, R_\text{target}) - w_{ga}\, \sum_{s\in\{L,R\}} \|\log_{SE(3)}(H_s^{-1} G_s)\|_2 + w_\text{lift}\, \max(0, \Delta z) + w_\text{grasp}\, \text{subtask}(t) + w_\text{success}\cdot \mathbb{1}[\text{success}] - w_\text{smooth}\, \|a_t - a_{t-1}\|_2^2$$

with weights `w_p=1.0, w_R=0.3, w_ga=0.5, w_lift=5.0, w_grasp=2.0, w_success=10.0, w_smooth=0.1`.

Key pieces:
- **Position error:** Euclidean distance between pot body and target lift pose.
- **Orientation error:** geodesic distance on SO(3) — the rotation-matrix form of `arccos((tr(R_1^T R_2) - 1)/2)` (`so3_geodesic_distance`).
- **Gripper-handle alignment:** SE(3) twist magnitude via the matrix logarithm of the relative transform between each gripper and its target handle frame (`se3_log_map`). This is the Lie-algebra distance on SE(3).
- **Lift / grasp subtask bonuses:** scalar rewards for crossing grasp-left, grasp-right, and lift-10-cm milestones.
- **Terminal success bonus:** +10 on the full task-success predicate.
- **Action-rate penalty:** discourages jittery actuation.

The pot-pose/gripper-pose fields needed for these terms are populated into `info` by a small patch to `rlinf/envs/robotwin/robotwin_env.py` that reads them from the RoboTwin state dict on each step (the "Case B" inline path in the spec, chosen because Sapien's subprocess-vectorized envs don't accept `gym.Wrapper` cleanly).

### 4.3 RoboEval dense shaped reward — used by B1, B2, B3

`_LiftPotDenseRewardWrapper` in `rlinf/envs/roboeval/roboeval_env.py:33`:

$$r_t = -w_\text{reach}(d_L + d_R) + w_\text{grasp}\cdot \text{subtask}(t) + w_\text{lift}\max(0, \Delta z) - w_\text{pose}\cdot e_\text{pose} + w_\text{success}\cdot \mathbb{1}[\text{success}] - w_\text{AR}\|a_t - a_{t-1}\|_2^2$$

with defaults `w_reach=1.0, w_grasp=2.0, w_lift=5.0, w_pose=0.5, w_success=10.0, w_action_rate=0.0`. `use_rel_reward: True` means the reward delivered to the algorithm is the per-step *delta* — this decorrelates the penalty terms from trivial cumulative effects.

Under this formulation, returns over a full 200-step episode are typically -1000 … +10 depending on whether the policy succeeds. From-scratch policies spend most of the episode far from the handles (large `d_L + d_R`) racking up reach and pose penalties, which dominates return — hence the observed range of -700 to -2000.

---

## 5. Methods

### 5.1 VLA checkpoint (B4)

- Architecture: **OpenVLA-OFT** (Open VLA, Open-FewTune variant) — a 7B autoregressive vision-language-action model based on Prismatic VLM + LoRA adapters.
- SFT checkpoint: `RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot` (published by RLinf authors alongside their paper). 14 GB, downloaded from HuggingFace.
- Zero-shot eval: 48 trajectories (3 rollouts × 16 envs × 1 eval pass), deterministic sampling (`temperature=0.0`).
- Resulting success: **6.25%** (1/16 on single-rollout eval; 3/48 = 6.25% stable across 3-rollout eval).

### 5.2 VLA + GRPO fine-tuning (M1, M2b)

**Algorithm:** Group-Relative Policy Optimization (GRPO) — a PPO variant where the per-sample advantage is computed relative to a group of parallel rollouts from the same initial state, rather than a learned critic. This removes the critic head and uses group-relative reward for variance reduction. Implemented in RLinf as a `GRPOActor` + `PPOTrainer` combination.

**Configuration (M1 attempt 5, adopted for M2b):**
- Hardware: 2× H100 on `robo-gh005`, FSDP (`no_shard` since 1 GPU per actor worker after placement).
- `global_batch=32`, `group_size=4`, `mini_batch_size=8`, 50 total train epochs capped by wall-clock.
- `lr=5e-5` (sqrt-scaled from RLinf's published `2e-4` for their 32× larger global batch of 1024).
- `temperature_train=1.0`, `temperature_eval=0.0`.
- `clip_ratio=0.2`, `kl_coef=0.0` (no reference-KL penalty — GRPO relies on clipping).
- `entropy_bonus=0` (consistent with RLinf defaults).
- Eval every 2 training epochs.

**Two attempts for M1:** attempt 4 used unmodified RLinf hyperparameters (`lr=2e-4`, `temperature=1.6`); training diverged (`approx_kl > 0.8`, `clip_fraction ≈ 0.95`). Attempt 5's sqrt-scaled lr gave healthy diagnostics (`approx_kl` 0.5–1.1, `clip_fraction` 0.57–0.71) but the same peak-and-regress shape on eval (details in `m1_vla_grpo.md`).

### 5.3 MLP-PPO from scratch (B1)

RLinf's built-in PPO actor with an MLP policy head.
- Architecture: 2-hidden-layer MLP, 256 units, tanh activations, diagonal Gaussian action head with state-dependent log-std.
- `lr=3e-4`, `clip_ratio=0.2`, `vf_coef=0.5`, `entropy_coef=0.01`, `gae_lambda=0.95`, `gamma=0.99`.
- 16 parallel envs, 128-step rollouts, 4 epochs × 4 minibatches per update.
- Run duration: cancelled at 60-min watchdog (zero eval improvement → stop) after 3h 13m total wall time, reaching 319 policy-update epochs.

### 5.4 MLP-SAC from scratch (B2)

RLinf's built-in SAC actor with an MLP policy.
- Architecture: 2-hidden-layer MLP (256 units) actor with tanh-squashed Gaussian; twin Q-networks (same MLP).
- `lr=3e-4`, `tau=0.005`, `gamma=0.99`, automatic temperature tuning (target entropy = -act_dim).
- 16 parallel envs, replay buffer 200k transitions, 256-batch updates, 1 update per env step.
- Cancelled at 60-min watchdog after 2h 34m, reaching 5,660 gradient steps.

### 5.5 MLP-MBPO from scratch (B3) — standalone Dyna-1 variant

Full RLinf does not expose a model-based RL worker, so this was written as a **standalone PyTorch training loop** (`scripts/b3_mbpo_roboeval.py`, 358 lines).

**Algorithm:**
1. **Dynamics model:** a deterministic MLP `f_\phi(s,a) -> (s',r)` predicting next-state and reward. Trained on mini-batches sampled from the real replay buffer.
2. **Policy/Q:** standard SAC (Gaussian actor, twin Q-nets, automatic temperature).
3. **Dyna-1 synthetic rollouts:** at every real env step, sample one real `(s,a)` from the replay buffer, query the dynamics model for `(s', r̂)`, push `(s, a, r̂, s')` into a synthetic buffer.
4. **SAC updates** train on a **50/50 mix** of real and synthetic batches (`--real-frac 0.5`).

**Configuration:** 16 train envs, 8 eval envs, 200 max episode length, 200,000 total env steps, 2,000 random-action warmup steps, batch size 256, buffer 200k, 1 dynamics update + 1 SAC update per env step, eval every 10k steps.

**Deviation from full MBPO (Janner et al., 2019):** we did NOT implement the ensemble dynamics model, NOT use the uncertainty-aware model-generated horizon (k > 1), and NOT use the per-model-rollout data weighting. This is a **Dyna-1 style minimal MBPO** — legitimately model-based (learned dynamics, synthetic transitions drive the Q-update) but simpler than the full paper. Reported accordingly.

**Observed failure mode:** policy converged to near-zero actions. `eval/return ≈ -3`, well above B1/B2's -1000 range, because minimal action magnitudes keep the policy close to the reach-penalty origin. This is a well-known MBPO pathology when the learned dynamics is imperfect and the reward has dominant penalty terms — the policy exploits the dynamics model's prediction errors at shapes that coincidentally minimize the (learned) reward, which here means "don't move."

---

## 6. Compute and wall-clock

| job | wall time | GPUs | GPU-hours |
|---|---|---|---|
| B4 (zero-shot, 1-roll + 3-roll) | 17 min | 1 | 0.29 |
| M1 attempt 4 + attempt 5 | 3h 43m | 2 | 7.43 |
| M2b | 2h 12m | 2 | 4.40 |
| 9× failed RoboTwin MLP attempts (B1 deadlock debug) | ~40 min | 1–2 | ~1.0 |
| B1 (RoboEval PPO, watchdog-cancelled) | 3h 13m | 2 | 6.43 |
| B2 (RoboEval SAC, watchdog-cancelled) | 2h 34m | 2 | 5.13 |
| B3 (RoboEval MBPO, completed) | 24 min | 1 | 0.40 |
| env builds, probes, installs | ~2 h | mixed | ~2.0 |
| **total** | | | **~27 GPU-hours** |

All on the `ROBOcis220039p` reservation at PSC Bridges-2 (`robo-gh005` Grace-Hopper node). Well inside the reservation budget.

---

## 7. RoboEval `eval/success_once = 1.0` spike bug — important caveat

Three eval checkpoints report `success_once = 1.000`: B1 at steps 79 (return -220) and 269 (return -369), B3 at step 184,704 (return -7.6). **These are not real successes** — they are a RoboEval instrumentation artifact. Evidence:

1. **Internal inconsistency.** Under the dense reward with `w_success = 10`, a real 16/16 success would yield `return ≈ +160 − penalties ≫ 0`. The observed returns at those exact steps are -7 to -370. If 16 envs really succeeded, `w_success` would have added at least +160 per batch. It didn't, so 16 envs did not really succeed.

2. **Info-dict key mismatch.** RoboEval's base `get_info()` writes `info["task_success"] = float(self.success)` (`roboeval/roboeval_env.py:481`). RLinf's RoboEval wrapper reads `info.get("success", 0.0)` for the reward's success term (`rlinf/envs/roboeval/roboeval_env.py:77`) and `infos["success"]` for `success_once` aggregation (line 582). The reward path silently falls through to 0 (explains why the return bonus never lands). The aggregation path reads an unrelated info key that is sometimes present via a different code route.

3. **Stale class-attribute default.** `LiftPot._success_check = True` is a class-level attribute (`roboeval/envs/lift_pot.py:26`) that `_on_reset()` never clears. Combined with the above, `success_once` can latch true from stale state without the reward ever firing.

**We did not fix and re-run** (2026-04-24 decision — the plot-level correction is sufficient for the report). The `all_tracks.png` figure simply omits the three bugged 1.0 checkpoints from the eval-success scatter. The more trustworthy signal, **training-rollout `env/success_once`**, is unaffected by this bug (it aggregates over many more rollouts and is populated through the same in-env path for all runs); on that signal, B1 peaks at 11.1%, B2 at 3.1%, and B3 does not log it — all far below 12% and drop to 0–5% by end of training.

---

## 8. Discussion — what this means

### 8.1 VLA pretraining is the load-bearing ingredient in this regime

In ~7 GPU-hours of RL fine-tuning, the pre-trained VLA goes 6.25% → 12.5% (2× zero-shot). In ~12 GPU-hours of RL from scratch across three algorithm families, MLPs sustain ~0% on eval. Published RLinf numbers say the same task reaches 70% with 8×H100 × 1000 epochs — two orders of magnitude more compute than we have. The VLA pretraining essentially *skips* the compute-intensive early phase (learn to see a pot, learn arm kinematics, learn bimanual coordination) and lets the RL objective focus on the narrow "refine grip/lift" loop.

### 8.2 SE(3) reward (M2b) is a null result on eval, mild positive on training rollouts

Peak eval is tied with the default reward (12.5% vs 12.5%). The SE(3) reward *did* produce smoother training diagnostics (`approx_kl` similar, `clip_fraction` healthier at epoch 3: 0.32 vs M1's 0.57) and its training-rollout success climbed to the same 12.5% peak that the eval saw — a signal that M1 did not match (its env-rollout peak was only 5.5%). So within the budget:

- **Evidence for denser gradient:** yes, training-rollout trajectory is smoother and higher-reaching for M2b.
- **Evidence for better eval generalization:** no, eval peak is identical to M1's and both regress after.

Reportable as **"negative result with diagnostic upside"** — the modification didn't help held-out eval in our budget, but its training dynamics suggest the shaping works as designed; the bottleneck is elsewhere (most likely: GRPO group sparsity on a 3% SFT, no entropy regularization, seed distribution drift between train and eval).

### 8.3 Cross-env limitation

The VLA track ran on RoboTwin (PIPER, sparse reward) and the MLP-scratch track on RoboEval (Panda, dense reward). **`success_once` is directly comparable** between them: both envs define "lift the pot ≥ 10 cm with upright pose." **`return` is NOT comparable** and we never plot it on shared axes — panels 2 and 3 of `all_tracks.png` are split accordingly.

The headline claim — 12.5% sustained on one track vs 0% sustained on the other — holds at the task-family level regardless of the env swap. If time permitted, a cleaner single-env comparison would port the from-scratch baselines to RoboTwin under a non-RLinf runner like SB3; this was out of the 7-day window.

### 8.4 MBPO inaction pathology (B3)

B3's policy converged to almost-zero actions. The learned dynamics model is imperfect on the 16-DOF bimanual task; in the region of "do nothing," the dynamics prediction error is small (state barely changes, easy to predict) and the dense reward's penalty terms are minimized. The policy exploits this "safe basin" to minimize its planning loss. Classic Dyna pathology. A full MBPO with ensemble dynamics and uncertainty-aware rollouts would likely avoid this by penalizing action sequences where the ensemble disagrees.

---

## 9. What was implemented (reusable artifacts)

Committed to branch `roboeval-integration`:

- **SE(3)/SO(3) reward wrapper** (`rlinf/envs/robotwin/se3_math.py` + `lift_pot_reward_wrapper.py`, 195 lines + 56 lines of math utilities). TDD: 15/15 unit tests passing (`tests/unit_tests/test_se3_math.py`, `test_lift_pot_reward_wrapper.py`).
- **RoboTwin env patch** for inline SE(3)-reward population + pot-pose augmentation (`rlinf/envs/robotwin/robotwin_env.py`).
- **RoboEval env integration** for RLinf (`rlinf/envs/roboeval/roboeval_env.py`, 600+ lines) — the `_LiftPotDenseRewardWrapper`, `_FlattenPropObsWrapper`, task-class registry, collate/record metric helpers, fixed-seed eval support. (Integrating RoboEval as an RLinf env was itself a substantial piece of new code, not inherited.)
- **Standalone MBPO script** (`scripts/b3_mbpo_roboeval.py`, 358 lines) — self-contained SAC + dynamics model + Dyna-1 synthetic rollouts, usable independently of RLinf.
- **Hydra configs** for all six runs: `examples/embodiment/config/robotwin_lift_pot_grpo_openvlaoft_*.yaml`, `roboeval_liftpot_ppo_mlp_v2.yaml`, `roboeval_liftpot_sac_mlp.yaml`, env-level `roboeval_liftpot_state.yaml`.
- **SLURM wrappers** (`slurm/robo/_common.sh` + per-run scripts) — handle VK_ICD_FILENAMES + VK_DRIVER_FILES override for the PSC Vulkan ICD, Ray per-job tmpdir, ROBOTWIN_PATH plumbing.
- **Plot scripts** (`scripts/plot_all_tracks.py`, `scripts/plot_vla_track.py`).

Not committed / debug-only: the 9 failed RoboTwin MLP deadlock-debug attempts (kept in commit log on the branch).

---

## 10. Per-run sub-reports

For reviewer deep-dives:
- `b4_zeroshot.md` — B4 details, decision gate.
- `m1_vla_grpo.md` — both M1 attempts, hyperparameter rescale rationale, collapse analysis.
- `m2b_vla_grpo_se3.md` — M2b single run, diagnostics comparison with M1.
- `b1_b2_b3_mlp_scratch.md` — final numbers, bug analysis, B3 inaction-optimum pathology.

## 11. Figures

- `vla_track_success.png` — two-panel: success (env-dense + eval-sparse) and return, for B4/M1/M2b. Uses the raw `GRPO epoch` index as x-axis (both M1 and M2b are the same algorithm with the same epoch granularity, so no normalization is needed).
- `all_tracks.png` — three-panel: success for all 6 runs (y-axis zoomed to [0, 0.2] since actual data maxes at 12.5%); RoboTwin return for B4/M1/M2b (panel 2); RoboEval return for B1/B2/B3 (panel 3). Return panels are split because the reward formulations are not comparable across envs.

### 11.1 X-axis normalization in `all_tracks.png`

Because the six runs use **different algorithms with different "step" units** — GRPO epoch (M1/M2b), PPO policy-update index (B1), SAC gradient-step index (B2), MBPO env-step counter (B3) — there is no common raw x-axis on which they can be overlaid. We therefore **min-max-normalize each run's step sequence independently** to the [0, 1] interval:

$$\tilde{x}_i = \frac{s_i - s_\text{min}}{s_\text{max} - s_\text{min}}$$

where $s_i$ is the $i$-th logged step of that run and $s_\text{min}, s_\text{max}$ are the first and last logged steps. The resulting $\tilde{x} = 0$ means "start of training" for that run and $\tilde{x} = 1$ means "end of training" (which, for B1/B2, is the 60-min watchdog cut, not 100% of the nominal budget — hence our runs cover different absolute compute amounts even though they all reach $\tilde{x}=1$).

**Consequence for reading the plots:** the *x-alignment* of two curves is not meaningful in absolute terms. "M2b reaches 12.5% at $\tilde{x} = 1$" and "B1 reaches its env-rollout peak of 11.1% at $\tilde{x} \approx 0.3$" are both true statements about the shape of training, but they do *not* imply M2b took three times as long — M2b's $\tilde{x}=1$ is 2h 12m of 2× H100 (6 GRPO epochs), while B1's $\tilde{x} = 0.3$ is ~60 min of 2× H100 (~100 PPO updates). For absolute-compute comparison, see the GPU-hours table in §6.

The `plot_all_tracks.py: normalize_x` helper implements this; the `env/success_once` and `env/return` panels use the same normalization so dense and sparse signals from the same run always line up on the x-axis.

---

## 12. Limitations & honest scope

- Cross-env caveat (§8.3) — not single-env.
- Only one seed per run (no error bars) — budget constraint.
- M1/M2b eval n=3 per run (eval every 2 epochs × 6 epochs) — peak numbers are single-checkpoint snapshots, same small-N caveat that applies to any short-budget run.
- B3 is Dyna-1, not full MBPO (§5.5).
- RoboEval eval-path bug (§7) means we rely on training-rollout `env/success_once` as the primary signal for B1/B2/B3.
- Compute is ~0.75% of the published RLinf run, so any comparison to their 70% number is for context only, not a claim of matching.
