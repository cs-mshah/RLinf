# Design: RL Fine-Tuning for a Pretrained VLA on RoboTwin LiftPot

**Course:** ROB 831 term project
**Date:** 2026-04-21
**Author:** @HJoonKwon (with Claude as collaborator)
**Status:** Draft — pending review

---

## 1. Problem & Contributions

The term project asks for (a) three RL-algorithm baselines and (b) two proposed modifications on top of those baselines. This spec defines how those pieces fit together for a **bimanual pot-lifting** manipulation task.

**Contributions:**
1. **Modification 1 — Pretrained VLA + RL.** Start from a publicly available VLA SFT'd on the task (`RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot`). Measure its zero-shot performance and compare to the same VLA after RL fine-tuning with GRPO (on-policy, PPO-family). Tests the hypothesis that RL on top of a pretrained generalist policy beats either ingredient alone.
2. **Modification 2 — Geometrically-principled reward shaping.** Replace RoboTwin's default orientation / pose terms with **SO(3) geodesic distance** and an **SE(3) log-map** twist norm. Compare success rate and sample efficiency of the three baselines with and without the new reward.

**Non-contributions (explicitly):**
- The environment (RoboTwin lift_pot) is not a contribution; we're using a community-standard bimanual manipulation task so that pretrained VLA checkpoints align with our embodiment. An earlier RoboEval integration (in PRs 1–3) is deliberately set aside.
- We do not train a new VLA from scratch. We do not implement classical MPC.

---

## 2. Environment & Task

**Task:** RoboTwin `lift_pot` — two PIPER arms grip handles of a kitchen pot on a table and lift it to a target pose.

**Embodiment:** Bimanual PIPER (14-DOF: 2× [6 arm joints + 1 gripper]).

**Why RoboTwin:**
- The only available pretrained VLA checkpoint for this pot-lifting task (`RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot`) is tied to the PIPER embodiment.
- RoboTwin is already integrated in RLinf (`rlinf/envs/robotwin/robotwin_env.py`) and has a working GRPO training config (`examples/embodiment/config/robotwin_lift_pot_grpo_openvlaoft.yaml`).
- The alternative (RoboEval LiftPot, 16-DOF BimanualPanda) would require SFT'ing a VLA from scratch — out of scope for a course project.

**Observation modes:**
- **RGB** — used by the VLA runs (head + 2 wrist cameras, matches the SFT checkpoint).
- **State** — used by the MLP baselines. *Not currently exposed by the RoboTwin env wrapper*; we need to add a state-obs mode (proprioception + pot pose + target pose) to `robotwin_env.py`. See §7.

**Action mode:** RoboTwin's default joint-position control at 25 Hz (matching the SFT distribution).

**Reward:** RoboTwin's shaped dense reward by default. For Modification 2 we wrap it with an SE(3)/SO(3)-based replacement (§6).

---

## 3. Experimental Matrix

| # | Name | Policy | Training | Obs | Purpose |
|---|------|--------|----------|-----|---------|
| **B1** | MLP-PPO | MLP (~256 hidden) | On-policy PPO | state | Baseline — on-policy model-free |
| **B2** | MLP-SAC | MLP | Off-policy SAC | state | Baseline — off-policy model-free |
| **B3** | MLP-MBPO | MLP (SAC actor) | SAC + learned-dynamics ensemble (Dyna) | state | Baseline — model-based |
| **B4** | VLA zero-shot | OpenVLA-OFT (SFT'd, frozen) | *none* | vision | Baseline — pretrained only, no RL |
| **M1** | VLA-GRPO | OpenVLA-OFT | GRPO fine-tune (on-policy) | vision | **Modification 1** — add RL to a pretrained VLA |
| **M2a** | MLP-SAC + SE(3) | MLP | SAC | state | **Modification 2** — reward ablation on B2 |
| **M2b** | VLA-GRPO + SE(3) | OpenVLA-OFT | GRPO | vision | **Modification 2** — reward ablation on M1 |

**Seeds:** 3 for MLP runs (B1, B2, B3, M2a); 1 for VLA runs (B4, M1, M2b). The VLA cap is for compute, not principle — we acknowledge the single-seed variance in the writeup.

**Key comparisons the plots will make:**
1. B1 vs B2 vs B3 — which RL family wins from-scratch on this task?
2. B4 vs M1 — does RL fine-tuning on top of a pretrained VLA help?
3. {B2, M1} vs {M2a, M2b} — does the SE(3)/SO(3) reward help, across policy classes?
4. Best-of-MLP vs M1 — is VLA+RL better than best from-scratch RL?

---

## 4. Algorithm designs

### 4.1 B1 — MLP-PPO (on-policy, from scratch)

- Policy: 2-layer MLP, hidden 256, tanh activation.
- Action head: independent-std Gaussian (shared with SAC), tanh squashing with per-joint bounds from the PIPER ctrlrange (analogous to the RoboEval PR 2 fix).
- Obs: state vector (proprioception 14 + pot 7 + target 7 + phase flags ≈ 30 dims).
- Algorithm: PPO-GAE (γ=0.99, λ=0.95, clip=0.1, 16 update epochs, entropy bonus 0.01). Based on the tuned v2 RoboEval config, not a direct port.
- Envs: 32 parallel train, 16 eval, horizon 200.
- Expected budget: ~2 h on 1× H100.

### 4.2 B2 — MLP-SAC (off-policy, from scratch)

- Same policy architecture as B1 but with dual Q-heads (`add_q_head=True, add_value_head=False`), auto-tuned entropy (target_entropy = -action_dim = -14), τ=0.005.
- Replay buffer size 1M transitions.
- Model-to-env update ratio: 20 SAC updates per env step (standard).
- `ignore_terminations=True` for infinite-horizon bootstrapping (RLinf convention).
- Expected budget: ~4 h on 1× H100.

### 4.3 B3 — MLP-MBPO (model-based, SAC + Dyna)

This is the only baseline that requires new RLinf infrastructure.

**Algorithm (Janner et al. 2019, MBPO):**
1. Vanilla SAC collects real transitions → `D_real`.
2. Every `M` env steps (500), retrain an **ensemble of N=5 MLP dynamics models** `f_φᵢ(s, a) → (Δs, r̂)` on `D_real`, with early-stopping on a held-out 10% split.
3. At each env step, generate **k=3 synthetic rollouts** from random start-states in `D_real`, using one randomly-picked ensemble member per rollout, store in `D_model`.
4. Train SAC on mixed batches: 95% from `D_model`, 5% from `D_real`.

**Why SAC as the base (not PPO):** Synthetic rollouts from a learned dynamics model are biased. Off-policy SAC tolerates bias because it reuses experience across many gradient steps without on-policy assumptions; PPO would need extra correction (importance weighting, trust-region adjustments) to handle model-generated data. SAC + Dyna is the standard pairing in the MBRL literature and stacks cleanly on top of B2 as a single-variable ablation.

**New files:**
- `rlinf/algorithms/mbpo/dynamics_ensemble.py` — ensemble MLP with `train_step`, `predict`, `compute_uncertainty`.
- `rlinf/workers/actor/mbpo_worker.py` (or adapter on `sac_worker`) — model-training hook, synthetic buffer, mixed sampling.
- `examples/embodiment/config/robotwin_lift_pot_mbpo_mlp.yaml`.

**Scope estimate:** ~400–500 LOC + config. 2–3 days.

**Risk:** SAC worker in RLinf may have assumptions (synchronous rollouts, specific replay interface) that resist a Dyna-style injection. Budget an audit of `sac_worker.py` before committing the interface. If the audit shows it's more invasive than expected (>1 week), fall back to a PPO-MBRL variant (document the deviation).

**Additional risk:** Learned dynamics on 14-DOF bimanual manipulation can be highly inaccurate — MBPO's success may be flat. A flat B3 curve is still a legitimate negative result for the course.

### 4.4 B4 — VLA zero-shot (no RL)

- Model: `RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot` (HuggingFace).
- Config: `robotwin_lift_pot_grpo_openvlaoft.yaml` with `runner.only_eval=True`.
- GPUs: 1× H100 (reduce from the 8-GPU FSDP config; only eval forward passes needed).
- Eval: 16 envs × 1 rollout, 200-step horizon, fixed reset seeds.
- Expected budget: ~30–60 min.

**Decision gate after B4:** if zero-shot success > 80%, M1's headroom is small and we reweight effort toward reward-modification experiments. If success is in the 20–70% band, M1 is the most interesting experiment.

### 4.5 M1 — VLA-GRPO (RL fine-tune, scaled to 1 GPU)

The existing `robotwin_lift_pot_grpo_openvlaoft.yaml` assumes 8-GPU FSDP with global batch 1024 and `group_size=8`. We rescale for a single H100:

| Parameter | 8-GPU default | 1-GPU config | Reason |
|-----------|---------------|--------------|--------|
| GPUs | 8 | 1 | reservation constraint |
| FSDP | yes | disabled (single-GPU) | no sharding needed |
| LoRA | `is_lora: True` | *kept* | ~0.2% trainable params, essential |
| `micro_batch_size` | 32 | 2 | fits H100 80GB with LoRA + gradient checkpointing |
| `global_batch_size` | 1024 | 32 (16× grad-accum × 2) | keep effective batch moderate |
| `group_size` | 8 | 4 | half the rollouts per GRPO update, less exploration breadth |
| `max_epochs` | 1000 | 150 | enough to see trajectory, fits wall-time |
| `val_check_interval` | 10 | 10 | unchanged |

- Training backend: switch from FSDP to plain single-GPU DDP/pure torch.
- Expected budget: ~8–12 h on 1× H100, 1 seed.
- **Convergence risk:** smaller effective batch + fewer rollouts per GRPO step may harm convergence. We report the result honestly even if it fails to improve over B4.
- If B4 is already >80% success, we reduce `max_epochs` further or skip M1 with documented reasoning.

### 4.6 M2a — MLP-SAC + SE(3)/SO(3) reward

- Identical to B2 except the env is wrapped with the new reward (§6).
- Paired comparison: run the same seeds as B2, same `max_env_steps`, so the only difference is the reward signal.

### 4.7 M2b — VLA-GRPO + SE(3)/SO(3) reward

- Identical to M1 except the env wrapper uses the new reward.
- The reward wrapper composes with the RGB observation pipeline (the reward is a scalar per step, doesn't affect obs).

---

## 5. Metrics

**Primary metric (reported in headline plot):**
- **Eval success rate** — `success_once` averaged over 16 eval envs, computed every 10 epochs.

**Task-phase diagnostics** — new signals we need to extract from RoboTwin's lift_pot task and surface in `infos`:
- `grasp_left_success` / `grasp_right_success` — binary, both handles gripped.
- `lift_distance` — vertical pot displacement from initial height (m).
- `pose_error_pos` — L2 position error of pot centroid vs target (m).
- `pose_error_rot` — SO(3) geodesic distance vs target (rad).
- `time_to_first_grasp` / `time_to_success` — episode step at which each happened; `NaN` if never.

**Behavioral diagnostics** (cheap, relevant to the reward story):
- `action_rate` — mean `‖aₜ − aₜ₋₁‖²` per episode.
- `action_energy` — mean `‖aₜ‖²` per episode.

**RL diagnostics** (already logged): `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `grad_norm`, `lr`.

**Reporting convention:**
- Final numbers = mean of the **last 3 eval checkpoints** (avoid cherry-picking a lucky spike).
- MLP runs: mean ± std across 3 seeds.
- VLA runs: single-seed number with a footnote on the variance limitation.

**Plots for the writeup:**
1. Success rate vs env steps — all 7 runs overlaid.
2. Time-to-success CDF — final checkpoints.
3. Reward-ablation bar chart — B2/M2a and M1/M2b paired.
4. Eval videos — qualitative inspection of smooth vs jerky behavior.

---

## 6. SE(3)/SO(3) Reward Design (Modification 2)

**Motivation:** RoboTwin's default reward uses Euclidean-style orientation terms (quaternion-diff or Euler-diff). These are not metrics on SO(3) and produce noisy, non-monotonic gradients in the physical angle. The hypothesis is that replacing them with Lie-group-correct quantities yields a cleaner learning signal.

**New file:** `rlinf/envs/robotwin/lift_pot_reward_wrapper.py`

**Term definitions:**

1. **SO(3) geodesic orientation error** for the pot:
   ```
   θ_geo(R, R_target) = arccos(½ · (tr(Rᵀ R_target) − 1))    ∈ [0, π]
   ```
   Bounded, monotonic in the actual physical rotation angle. Computed via `scipy.spatial.transform.Rotation`.

2. **SE(3) pose cost** — combined position + orientation:
   ```
   c_SE(3)(T, T_target) = w_p · ‖p − p_target‖ + w_R · θ_geo(R, R_target)
   ```
   Weights tuned so that 1 cm error ≈ 3–5° error in penalty magnitude.

3. **Gripper-to-handle SE(3) alignment** (optional, if pose info is available) — penalize `‖ξ‖` where `ξ ∈ se(3)` is the log-map twist between gripper and handle frames. Encourages coordinated 6-DoF approach.

4. **Twist-space action smoothness** (if EE control is configurable) — replace joint-space smoothness with end-effector twist smoothness. For our joint-controlled setup, keep joint-space smoothness as a fallback.

**Composite reward:**
```
r_SE(3) = - w_p  · ‖p_pot − p_target‖
         - w_R  · θ_geo(R_pot, R_target)
         - w_ga · (‖ξ_left_gripper‖ + ‖ξ_right_gripper‖)   # optional
         + w_lift · max(0, lift_distance)
         + w_grasp · subtask_progress
         + w_success · 1[success]
         - w_smooth · ‖aₜ − aₜ₋₁‖²
```

**Config knobs** (in env config YAML): `w_p`, `w_R`, `w_ga`, `w_lift`, `w_grasp`, `w_success`, `w_smooth`, plus `reward_variant: {default, se3}` to switch.

**Default weights** (to tune empirically in Phase 3):
`w_p=1.0, w_R=0.3, w_ga=0.5, w_lift=5.0, w_grasp=2.0, w_success=10.0, w_smooth=0.1`.

**Implementation scope:** ~150–200 LOC + config knobs. 1 day.

**Risk:** RoboTwin may not expose pot's 6D pose or target pose cleanly through its gym interface. If we have to reach into the sim object directly, that's acceptable but fragile — document the access pattern.

---

## 7. Repo layout changes

| File | Change |
|------|--------|
| `rlinf/envs/robotwin/robotwin_env.py` | **Modify** — add `obs_mode: state` with proprioception + pot pose + target pose vector. Expose phase-level diagnostics in `infos`. |
| `rlinf/envs/robotwin/lift_pot_reward_wrapper.py` | **New** — SE(3)/SO(3) reward wrapper (Modification 2). |
| `rlinf/algorithms/mbpo/dynamics_ensemble.py` | **New** — MLP dynamics ensemble for B3. |
| `rlinf/workers/actor/mbpo_worker.py` | **New** — SAC + Dyna training loop. |
| `examples/embodiment/config/env/robotwin_lift_pot_state.yaml` | **New** — state-obs env config for MLP baselines. |
| `examples/embodiment/config/robotwin_lift_pot_ppo_mlp.yaml` | **New** — B1 config. |
| `examples/embodiment/config/robotwin_lift_pot_sac_mlp.yaml` | **New** — B2 config. |
| `examples/embodiment/config/robotwin_lift_pot_mbpo_mlp.yaml` | **New** — B3 config. |
| `examples/embodiment/config/robotwin_lift_pot_grpo_openvlaoft_1gpu.yaml` | **New** — M1 scaled-down VLA config. |
| `slurm/robo/` (directory) | **New** — per-run SLURM scripts targeting `--reservation=ROBOcis220039p`. |
| `docs/rob831-project/PRs/` | **Extend** — one PR note per new feature, matching the existing convention. |

---

## 8. Environment setup

**Target environment name:** `rlinf-openvlaoft` (conda), separate from collaborator's `roboeval` env.

**Setup sequence on PSC Bridges-2:**
```bash
module load anaconda3 cuda
conda create -n rlinf-openvlaoft python=3.11 -y
conda activate rlinf-openvlaoft

# From RLinf root:
bash requirements/install.sh embodied --model openvla-oft --env robotwin --venv .venv-openvlaoft
# OR: pip install the openvla-oft + robotwin requirements manually if the uv flow misbehaves on conda.

# RoboTwin asset download:
bash requirements/download_assets.sh  # adjust per RoboTwin instructions
```

**Fallback if conda + install.sh conflict:** use the uv flow directly (`bash requirements/install.sh ...` without conda), or the docker/apptainer image with `BUILD_TARGET=embodied-robotwin`.

**Env variables to set in all SLURM jobs:**
```
MUJOCO_GL=egl
EMBODIED_PATH=examples/embodiment/
REPO_PATH=$(pwd)
ROBOT_PLATFORM=ALOHA          # or whichever matches the SFT unnorm_key
HYDRA_FULL_ERROR=1
```

---

## 9. SLURM plan

**Reservations (yours):**
- `ROBOcis220039p`: `robo-gh005`, 8× H100, active now through 2026-05-15.
- `ROBOGPU`: `robo-gh006`, 8× H100, activates 2026-04-24, through 2026-05-24.

**Constraint:** sequential submission only — one job at a time on the reservation node.

**Pattern:**
```bash
# Submit phase-1 job, capture job ID
JOB1=$(sbatch --parsable slurm/robo/phase1_b4_eval.sh)
# Queue phase-2 after phase-1 succeeds
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/robo/phase2_b1_ppo.sh)
# etc.
```

**Per-run SLURM shape** (MLP baseline example):
```
#SBATCH --reservation=ROBOcis220039p
#SBATCH --partition=ROBO
#SBATCH --nodelist=robo-gh005
#SBATCH --gres=gpu:h100:1
#SBATCH --time=06:00:00
#SBATCH --job-name=b1_ppo
```

VLA runs get `--gres=gpu:h100:1` too (with the scaled-down config) and `--time=12:00:00`.

**Checkpointing**: `save_interval=10`, results under `../results/<job-name>_<jobid>/`. VLA saves LoRA adapter only (small files, fast).

---

## 10. Phased plan with decision gates

### Phase 1 — "First, get B4 working" (1–2 days)
1. Build `rlinf-openvlaoft` conda env.
2. Download `RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot` checkpoint.
3. Run B4 (eval only, 1× H100, ~30 min).
4. Sanity checks: success rate > 0, eval videos sensible, metrics in tensorboard.

**Decision gate:**
- B4 success > 80% → M1 headroom is small; reweight toward M2 experiments.
- B4 success ∈ [20%, 80%] → M1 is the most interesting run; proceed as planned.
- B4 success < 20% → investigate checkpoint / unnorm key / camera config before committing further compute. This is a debug gate, not a plan continuation.

### Phase 2 — "MLP baselines" (2–3 days)
5. Implement state-obs mode in RoboTwin env wrapper (§7).
6. Write B1/B2 configs (fresh, not ports).
7. Run B1 × 3 seeds, B2 × 3 seeds.

**Decision gate:** if both B1 and B2 are stuck at ≤5% success after 2× budgeted training time, the state features are probably inadequate — revisit §2's obs spec before moving on.

### Phase 3 — "The project contributions" (1–2 weeks)
8. Implement SE(3)/SO(3) reward wrapper (§6).
9. Run M2a (× 3 seeds) — pair with B2.
10. Implement MBPO (§4.3). Budget checkpoint: 1 week max. If the SAC-worker audit shows MBPO needs >1 week of refactoring, fall back to documenting "MBRL on VLA fine-tuning is out of scope" and drop B3.
11. Run B3 × 3 seeds.
12. Run M1 × 1 seed (scaled-down 1-GPU VLA-GRPO).
13. Run M2b × 1 seed.

### Phase 4 — Writeup
14. Generate all plots via `scripts/plot_return.py`.
15. Record eval videos for each run's final checkpoint.
16. Write the project report referencing `docs/rob831-project/PRs/` for implementation notes.

---

## 11. Risks & explicit descope points

- **MBPO risk (B3):** if `sac_worker.py` is not Dyna-friendly, fall back to PPO-MBRL variant or drop B3 and document why. Budget 1 week max for the refactor.
- **VLA convergence risk (M1):** single-GPU LoRA with small effective batch may fail to improve over B4. Honest report of the failure is acceptable for the course; also offer an "ideal M1 on 8 GPUs" as future work.
- **Pot pose access risk:** if RoboTwin doesn't surface the pot pose cleanly, §6 reward wrapper has to reach into the sim — document the access path.
- **Reservation conflict risk:** `ROBOcis220039p` ends 2026-05-15. If the project deadline is before then, all work fits. If deadline slips past 2026-05-24, secure a new reservation.

**Hard descope order if time is tight** (drop from the bottom):
1. M2b (VLA × SE(3) reward) — keep the VLA+RL result and MLP reward result, drop the combined cell.
2. B3 (MBPO) — drop if the infra work is too heavy.
3. A seed (MLP runs 3 → 2).
4. M1 seed count is already 1; no further reduction.

---

## 12. Success criteria for the project

- All 4 baselines report a success-rate curve, even if some are flat.
- M1 reports a single-seed curve comparing VLA zero-shot to VLA+GRPO.
- M2 reports a paired comparison of default vs SE(3) reward on at least MLP-SAC.
- The writeup discusses negative results honestly where they occur (e.g., if MBPO fails to learn on this task, that's fine — we explain why, using learned-model accuracy as a diagnostic).
