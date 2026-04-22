# Design: RL Fine-Tuning for a Pretrained VLA on RoboTwin LiftPot

**Course:** ROB 831 term project
**Date:** 2026-04-21
**Due date:** 2026-04-28 (hard — 7 days)
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

**Seeds:** 1 per run, across both MLP and VLA rows. Chosen for time/compute economy given a term-project timeline; we acknowledge the single-seed variance honestly in the writeup rather than inflating the seed count for performative confidence intervals we wouldn't trust.

**Key comparisons the plots will make:**
1. B1 vs B2 vs B3 — which RL family wins from-scratch on this task?
2. B4 vs M1 — does RL fine-tuning on top of a pretrained VLA help?
3. {B2, M1} vs {M2a, M2b} — does the SE(3)/SO(3) reward help, across policy classes?
4. Best-of-MLP vs M1 — is VLA+RL better than best from-scratch RL?

---

## 4. Algorithm designs

### 4.1 B1 — MLP-PPO (on-policy, from scratch)

**Algorithmic choice note (vs M1 = VLA-GRPO):** B1 uses PPO-clip with a learned value function + GAE; M1 uses GRPO with group-relative advantages. Both are in the PPO-clip family (same surrogate objective). They are not identical, but they are the *best-practice on-policy method for each policy class*: PPO's value-baseline advantage is more robust to the bootstrap phase of an MLP trained from random init (where a GRPO group of 4 rollouts would be uniformly ~0 reward → zero advantage signal), while GRPO's group-advantage shines on a pretrained VLA where rollout outcomes are varied. The writeup frames this as "best on-policy method per architecture" rather than strict algorithmic equivalence.


- **Do not port RoboEval configs.** Instead, follow the conventions RLinf already uses for RL fine-tuning on RoboTwin / other embodied envs (obs dict format, action pre/post-processing, normalization, dimension ordering). The RoboEval integration in PRs 1–3 had bugs we don't need to inherit.
- Policy: `rlinf/models/embodiment/mlp_policy` with the RLinf-standard construction (hidden dims from config, independent-std Gaussian, optional tanh squashing).
- Action bounds: read from the PIPER ctrlrange exposed by the RoboTwin env (whatever RLinf's existing RoboTwin env adapter provides — don't hand-roll them).
- Obs: state mode to be defined by §7 work; follow `rlinf/envs/<existing-state-env>` structure.
- Algorithm knobs (starting point, tune in Phase 2): PPO-GAE γ=0.99, λ=0.95, clip ratio 0.1–0.2, update epochs 8–16, entropy bonus 0.01. Exact values follow RLinf's other PPO-on-state configs (e.g. `frankasim_ppo_mlp.yaml`) rather than RoboEval.
- Envs: 32 parallel train, 16 eval, horizon 200. Adjust if the H100 allows more parallel envs without OOM.
- Expected budget: ~2 h on 1× H100.

### 4.2 B2 — MLP-SAC (off-policy, from scratch)

- Same principle: base on RLinf's existing SAC configs (e.g., `maniskill_sac_mlp.yaml` or `frankasim_sac_*.yaml`), not RoboEval.
- Policy architecture: RLinf's MLP policy with `add_q_head=True, add_value_head=False`, auto-tuned entropy (`alpha_type: softplus`, `target_entropy = -action_dim`), τ=0.005.
- Replay buffer, update ratio, `ignore_terminations`: use RLinf's SAC defaults.
- Expected budget: ~4 h on 1× H100.

### 4.3 B3 — MLP-MBPO (minimal Dyna-1, 7-day scope)

This is the only baseline that requires new RLinf infrastructure. Given the 7-day deadline we commit to a **minimal viable MBPO**: single dynamics model, horizon-1 synthetic rollouts (Dyna-1). The full MBPO (ensemble of 5, horizon-3 rollouts) is left as a stretch goal.

**Minimal algorithm:**
1. Vanilla SAC collects real transitions → `D_real`.
2. **Single** MLP dynamics model `f_φ(s, a) → (Δs, r̂)` — 2-layer MLP, hidden 256. Retrain every 500 env steps on `D_real`, early-stopping on a 10% held-out split.
3. At each env step, generate **one 1-step synthetic transition** (Dyna-1) from a random real state; store in `D_model`.
4. Train SAC on mixed batches: **95% from `D_model`, 5% from `D_real`** (Janner et al. 2019 §5; large model share is intentional — SAC's off-policy robustness absorbs model bias while real samples keep the Q-function grounded).

**Why SAC (not PPO) as the base:** Synthetic rollouts from a learned dynamics model are biased. Off-policy SAC tolerates bias because it reuses experience across many gradient steps without on-policy assumptions; PPO would need extra correction (importance weighting, trust-region adjustments). SAC + Dyna is the standard pairing in MBRL literature and stacks cleanly on top of B2 as a single-variable ablation.

**Stretch upgrade (post-Day-4 if schedule allows):** add ensemble (N=5) and grow horizon to H=3 for a true MBPO.

**New files:**
- `rlinf/algorithms/mbpo/dynamics_model.py` — single MLP dynamics with `train_step`, `predict` (API leaves room for ensemble extension).
- **Modified:** the SAC worker — model-training hook, synthetic buffer, mixed sampling. Exact file located during Day 4 morning audit.
- `examples/embodiment/config/robotwin_lift_pot_mbpo_mlp.yaml`.

**Scope estimate:** ~200 LOC + config. 1 day including the SAC worker audit.

**Risk (hard):** SAC worker in RLinf may have assumptions (synchronous rollouts, specific replay interface) that resist Dyna-style injection. **Mitigation:** if the Day-4-morning audit suggests the injection is >1 day of work, **drop B3 entirely** and document the descope. The project still has 2 RL baselines (PPO, SAC) — we'd note that MBRL requires infrastructure beyond 7-day scope.

**Additional risk:** Learned dynamics on 14-DOF bimanual manipulation with a single (non-ensemble) model will be inaccurate; MBPO's success rate may be flat or worse than B2. A flat B3 curve is still a legitimate negative result — we report it honestly.

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

- Training backend: single-process, no FSDP, no DDP (DDP on a single GPU is a no-op).
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
- All runs: single-seed number, with a footnote on the variance limitation. No seed-averaged error bars; instead we report per-eval variance across the 16 parallel eval envs as a within-run variability proxy.

**Note on `pose_error_rot` as both metric and reward term:** `pose_error_rot` (SO(3) geodesic) appears here as a diagnostic and in §6 as part of the SE(3) reward. For the M2 comparisons this creates a Goodhart-like situation (we'd expect lower `pose_error_rot` on the runs whose reward includes it). That is the *point* of the reward ablation — the headline metric remains success rate, and `pose_error_rot` should be read as a mechanism-level diagnostic, not a fairness metric.

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

**Decisions on the optional terms:**
- **Gripper-to-handle SE(3) alignment** (term 3): **include in the default SE(3) variant**. It's the most interesting geometric contribution; if the pose info isn't available for handles, we fall back to gripper-to-pot-centroid (which always exists).
- **Twist-space action smoothness** (term 4): **exclude from the default**. Our control space is joint position, so joint-space smoothness remains the natural choice; twist-space smoothness is a follow-up idea for EE-control runs we aren't doing.

**`subtask_progress` definition:** scalar in [0, 1] computed as `0.5 · (grasp_left_success + grasp_right_success) + 0.25 · 1[lift_distance > 5 cm]` — i.e., 0 before any grasp, 0.5 at single grasp, 1.0 once both handles are gripped and the pot is lifted 5 cm. This is computed inside the reward wrapper from the same phase-level signals exposed in §5.

**Composite reward (SE(3) variant, `reward_variant: se3`):**
```
r_SE(3) = - w_p  · ‖p_pot − p_target‖
         - w_R  · θ_geo(R_pot, R_target)
         - w_ga · (‖ξ_left_gripper_to_handle‖ + ‖ξ_right_gripper_to_handle‖)
         + w_lift · max(0, lift_distance)
         + w_grasp · subtask_progress
         + w_success · 1[success]
         - w_smooth · ‖aₜ − aₜ₋₁‖²
```

**Config knobs** (in env config YAML): `w_p`, `w_R`, `w_ga`, `w_lift`, `w_grasp`, `w_success`, `w_smooth`, plus `reward_variant: {default, se3}` to switch.

**Default weights** (starting point, tune if needed):
`w_p=1.0, w_R=0.3, w_ga=0.5, w_lift=5.0, w_grasp=2.0, w_success=10.0, w_smooth=0.1`.

**Implementation scope:** ~150–200 LOC + config knobs. 1 day.

**Risk:** RoboTwin may not expose pot's 6D pose or target pose cleanly through its gym interface. If we have to reach into the sim object directly, that's acceptable but fragile — document the access pattern.

---

## 7. Repo layout changes

| File | Change |
|------|--------|
| `rlinf/envs/robotwin/robotwin_env.py` | **Modify** — add `obs_mode: state` with proprioception + pot pose + target pose vector. Expose phase-level diagnostics in `infos`. |
| `rlinf/envs/robotwin/lift_pot_reward_wrapper.py` | **New** — SE(3)/SO(3) reward wrapper (Modification 2). |
| `rlinf/algorithms/mbpo/dynamics_model.py` | **New** — single MLP dynamics model for B3 (minimal Dyna-1 variant). |
| SAC worker file (located during Day-4 audit) | **Modify** — add dynamics training hook + synthetic buffer + mixed-batch sampling. |
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

## 9. Compute budget & SLURM plan

**Compute budget** (1 seed per run, H100 80GB):

| Run | GPUs | Wall time |
|-----|------|-----------|
| B1 MLP-PPO | 1 | ~2 h |
| B2 MLP-SAC | 1 | ~4 h |
| B3 MLP-MBPO | 1 | ~6 h |
| B4 VLA zero-shot | 1 | ~0.5 h |
| M1 VLA-GRPO (1-GPU LoRA) | 1 | ~8–12 h |
| M2a MLP-SAC + SE(3) | 1 | ~4 h |
| M2b VLA-GRPO + SE(3) | 1 | ~8–12 h |
| **Total (sequential)** | | **~32–40 h** |

Total wall-clock is comfortably inside the deadline. The binding constraint is implementation/debugging throughput, not compute — we overlap coding with training runtime (write next config while current run trains), and execute stages ASAP rather than on a fixed day schedule. Opportunistic multi-GPU bursts (if an idle H100 becomes available) can further compress the VLA runs.

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

## 10. Execution plan (ordered stages, execute ASAP)

Deadline is 2026-04-28 but we execute as fast as the compute allows. No hard day-boxing. Implementation work is overlapped with training runtime wherever possible (write next config / wrapper / MBPO code while the current training job runs). If additional idle GPUs become available opportunistically (other ROBO nodes, non-reserved `GPU` partition, etc.), we use them to parallelize independent training jobs — but the default assumption is single-GPU sequential on `robo-gh005`.

### Stage 0 — Preflight
- Verify SLURM partition / reservation (`sinfo -p ROBO`, `scontrol show reservation ROBOcis220039p`).
- Build `rlinf-openvlaoft` conda env. Download SFT checkpoint.

### Stage 1 — B4 (VLA zero-shot eval)
- Run B4 (~0.5 h). Primary purpose: confirm env + checkpoint + eval pipeline all work end-to-end before committing any more compute.
- **Gate:** if B4 success < 20%, stop and debug checkpoint / `unnorm_key` / camera config. Otherwise proceed — the zero-shot number itself is useful regardless of magnitude.

### Stage 2 — M1 (VLA-GRPO fine-tune) — *the headline contribution, done early*
- Write the 1-GPU scaled-down VLA-GRPO config (§4.5).
- Run M1 (~8–12 h on 1 GPU). If an additional H100 is available opportunistically, 2 GPUs halves the wall time — worth checking at submission time.
- While M1 trains: **start implementing the SE(3)/SO(3) reward wrapper (§6)** — it's env-side and policy-agnostic, so its development is independent of M1.
- **Gate:** if M1 fails to improve over B4 at all (e.g., eval curve is flat or regressing after ~30% of `max_epochs`), pause and sanity-check: sampling temperature, grad-accum arithmetic, LoRA freeze/unfreeze layers. The single-GPU scaling is new and carries convergence risk.

### Stage 3 — M2b (VLA + SE(3) reward)
- Once the SE(3) reward wrapper from Stage 2 is done: run M2b (~8–12 h). Paired ablation against M1 — same model, same config, only the reward differs.
- **If M1 already produced a solid result (B4 success + substantial delta from M1)**, M2b's upside is adding SE(3) on top of an already-working VLA; if M1 was flat, M2b's purpose shifts to "does better shaping rescue a marginal run."

### Stage 4 — State-obs plumbing + B1 (MLP-PPO)
- Add `obs_mode: state` to the RoboTwin env wrapper (§7): proprioception + pot pose + target pose, expose phase-level diagnostics in `infos`.
- Write B1 config using RLinf's state-based RL conventions (not a RoboEval port).
- Run B1 (~2 h).
- **Gate:** if B1 is at ≤5% success after 2× budgeted training time, state features are likely inadequate — revisit the obs spec before continuing.

### Stage 5 — B2 (MLP-SAC)
- Write B2 config (reuse the state-obs env from Stage 4).
- Run B2 (~4 h).

### Stage 6 — M2a (MLP-SAC + SE(3) reward)
- SE(3) reward wrapper already exists from Stage 2, so M2a is cheap — just a new env config + job submission.
- Run M2a (~4 h). Paired ablation against B2.

### Stage 7 — MBPO audit + B3 (conditional)
- Audit the SAC worker to scope the Dyna-1 injection. **Hard gate:** if audit suggests >1 day of implementation, **drop B3** and document. Course still has two RL algorithms (PPO, SAC) plus the VLA experiments.
- If audit cleared: implement minimal Dyna-1 MBPO (§4.3), ~200 LOC. Run B3 (~6 h).

### Stage 8 — Analysis & writeup
- Generate plots (`scripts/plot_return.py`), record eval videos.
- Write the report (intro, methods, results, discussion of negatives, conclusion). Commit under `docs/rob831-project/`.

**Stage ordering rationale:** B4 → M1 → M2b → (MLP baselines) → M2a → B3. The VLA track is the headline contribution and carries the most unknown (single-GPU scaling of a bimanual OpenVLA-OFT GRPO config is not a well-trodden recipe) — doing it early means we find out quickly whether the approach works at all. The SE(3) reward wrapper is written during M1's training wall-clock, so by Stage 3 we can run M2b immediately; the same wrapper is reused for M2a with no additional code work. MLP baselines (B1, B2, MBPO) are cheap and predictable, so they land later without much schedule risk.

**Hard dropouts in priority order** (if the calendar gets tight):
1. **B3 (MBPO)** — first to drop if the Stage-7 audit shows Dyna-1 integration is >1 day. Course survives with 2 RL algorithms + clear scope justification.
2. **M2a** — second to drop; M2b already makes the SE(3)-reward point on the stronger VLA policy, so the MLP version is an ablation-of-an-ablation.
3. **B1 or B2** — very unlikely to drop (each is ≤4 h of compute and satisfies the RL-family rubric). Only drop if we genuinely can't finish MLP infrastructure in time.
4. **M1 and M2b** — never drop; these are the headline VLA contributions.

**Opportunistic parallelism:** if you pick up idle H100s outside the reservation:
- While M1 runs on the ROBO node, a spare GPU can start M2b on the same config+reward-wrapper as soon as the wrapper is written (independent training jobs).
- In the MLP phase (Stages 4–7), B1 / B2 / B3 / M2a are all independent and fully parallelizable across whatever GPUs are available.
Don't chase this aggressively; the sequential plan already fits the deadline. It's a bonus if idle capacity shows up.

---

## 11. Risks & explicit descope points

- **MBPO risk (B3):** if the Day-4 SAC-worker audit shows Dyna-style injection is >1 day of work, drop B3 and document. Course deliverable survives with 2 RL algorithms + clear justification.
- **VLA convergence risk (M1):** single-GPU LoRA with small effective batch may fail to improve over B4. Honest report of the failure is acceptable for the course; also offer an "ideal M1 on 8 GPUs" as future work.
- **Pot pose access risk:** if RoboTwin doesn't surface the pot pose cleanly, §6 reward wrapper has to reach into the sim — document the access path.
- **Timeline risk:** 7 days is tight. Every decision gate in §10 prioritizes shipping working experiments over completeness.
- **Reservation:** `ROBOcis220039p` is active through 2026-05-15, well past the 2026-04-28 deadline — no reservation-availability risk.

**Hard descope order if time is tight** (drop from the bottom):
1. M2b (VLA × SE(3) reward) — keep the VLA+RL result and MLP reward result, drop the combined cell.
2. B3 (MBPO) — drop if the Phase 2 audit says the infra work is too heavy.
3. M1 (VLA+GRPO fine-tune) — last to drop since it's the headline Modification 1 experiment.

(We already commit to 1 seed per run, so there's no further seed reduction available.)

---

## 12. Success criteria for the project

- All 4 baselines report a success-rate curve, even if some are flat.
- M1 reports a single-seed curve comparing VLA zero-shot to VLA+GRPO.
- M2 reports a paired comparison of default vs SE(3) reward on at least MLP-SAC.
- The writeup discusses negative results honestly where they occur (e.g., if MBPO fails to learn on this task, that's fine — we explain why, using learned-model accuracy as a diagnostic).
