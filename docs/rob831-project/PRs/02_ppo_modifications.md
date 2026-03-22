# PR: PPO Training Improvements for RoboEval LiftPot

**Source branch:** `ppo-modifications`
**Target branch:** `roboeval-integration`

---

## Summary

Fix critical training issues and add infrastructure for PPO training on the RoboEval LiftPot task. The main change is adding **tanh action squashing** to the MLP policy to keep actions within the environment's joint limits, fixing a mismatch between policy outputs and the action space that prevented effective learning. Also adds CNN policy config, video recording support, SLURM job infrastructure, and training metrics documentation.

## Key Changes

### Bug Fix: Tanh Action Squashing (critical)

**Problem:** The MLP policy sampled actions from an unbounded Gaussian (`Normal(mean, std)`) and passed them directly to the environment. The RoboEval env has bounded action spaces (`[-2.5, 2.5]` for 14 joints, `[0, 1]` for 2 grippers). Out-of-bounds actions were silently clipped by the env, but the policy's log-probabilities were computed on the *unclipped* actions — creating a train/inference mismatch that prevented learning. This also produced ~488MB of warning logs per run.

**Fix:**
- `rlinf/models/embodiment/mlp_policy/mlp_policy.py`:
  - Added `action_low`/`action_high` parameters to `MLPPolicy.__init__`
  - When bounds are provided, enables `tanh` squashing: `action = tanh(raw) * scale + bias` with per-dimension scale/bias buffers
  - `_generate_actions`: already had tanh path (used by SAC) — now activates for PPO when bounds are set
  - `default_forward`: added inverse-tanh (atanh) to recover raw actions before computing log-probs, with the correct Jacobian correction term
  - `_sample_actions`: fixed `logstd_range` clamping to only apply for network-output std (SAC), not independent learnable std (PPO)
- `rlinf/models/embodiment/mlp_policy/__init__.py`: forwards `action_low`/`action_high` from config
- `examples/embodiment/config/roboeval_liftpot_ppo_mlp.yaml`: added action bounds matching the env

All changes are backward-compatible — existing configs without `action_low`/`action_high` behave identically to before.

### Config Tuning

- `roboeval_liftpot_ppo_mlp.yaml`:
  - `total_num_envs`: 128 → 32 (fix OOM on V100-32GB)
  - `obs_dim`: 30 → 38 (correct observation dimension)
  - `micro_batch_size`: 256 → 32, `global_batch_size`: 256 → 64
  - `val_check_interval`: 50 → 10 (more frequent eval)
  - `component_placement`: single GPU → `"0,1"` (use both V100s)
  - Disabled train video recording (save compute), enabled eval video with `video_record_interval`

### New: CNN Policy Config

- `examples/embodiment/config/roboeval_liftpot_ppo_cnn.yaml` — full config for ResNet-based vision policy
- `examples/embodiment/config/env/roboeval_liftpot_vision.yaml` — RGB observation mode with 3 cameras (head, left_wrist, right_wrist) at 128x128

### Video Recording Support

- `rlinf/envs/roboeval/roboeval_env.py`:
  - Added video rendering in state mode: when `save_video=True`, attaches a camera and renders frames via `env.render()` even in state observation mode
  - Passes `render_mode="rgb_array"` to underlying RoboEval env when video is enabled
  - `_wrap_obs` now accepts `env_idx` for per-env rendering
- `rlinf/workers/env/env_worker.py`:
  - Added `video_record_interval` support — only flushes video every N rollouts to reduce I/O
  - Clears render buffers on skipped intervals to prevent memory growth

### SLURM Infrastructure

- `slurm/train.sh` — main training script with per-experiment results directory (`results/<job-name>_<job-id>/`)
- `slurm/wrapper.sh` — job submission wrapper that sets `LOG_DIR` and creates log directories
- `slurm/submit_job_test.sh` — lightweight test job for verifying cluster setup
- `.gitignore` — added `slurm/logs/`, `*.err`, `*.out`

### Plotting & Documentation

- `scripts/plot_return.py` — CLI script to plot mean return vs env steps from TensorBoard logs
- `docs/rob831-project/tensorboard-metrics.md` — reference doc explaining all TensorBoard metrics, results directory structure, and step conversion formula

## Files Changed

| File | Change |
|------|--------|
| `rlinf/models/embodiment/mlp_policy/mlp_policy.py` | Tanh squashing + log-prob correction for PPO |
| `rlinf/models/embodiment/mlp_policy/__init__.py` | Pass action bounds from config |
| `rlinf/envs/roboeval/roboeval_env.py` | Video recording in state mode |
| `rlinf/workers/env/env_worker.py` | Video record interval support |
| `examples/embodiment/config/roboeval_liftpot_ppo_mlp.yaml` | Config fixes + action bounds |
| `examples/embodiment/config/roboeval_liftpot_ppo_cnn.yaml` | New CNN policy config |
| `examples/embodiment/config/env/roboeval_liftpot_state.yaml` | Added video_record_interval |
| `examples/embodiment/config/env/roboeval_liftpot_vision.yaml` | New vision env config |
| `scripts/plot_return.py` | New plotting script |
| `docs/rob831-project/tensorboard-metrics.md` | New metrics documentation |
| `slurm/train.sh` | New training SLURM script |
| `slurm/wrapper.sh` | New job submission wrapper |
| `slurm/submit_job_test.sh` | New test job script |
| `.gitignore` | Ignore SLURM logs |

## Testing Done

- Model unit test: verified tanh-squashed actions are always within `[action_low, action_high]` bounds
- Log-prob computation: verified finite log-probs with correct Jacobian correction
- Backward compatibility: verified existing configs (no bounds, SAC with q_head) still work identically
- Previous run (without fix): flat return curve at ~-4.5, 488MB of OOB warnings — confirms the issue

## Known Issues / TODOs

- [ ] Run full training with tanh squashing and verify learning curve improves
- [ ] CNN policy config untested end-to-end (needs ResNet pretrained weights)
- [ ] `EMBODIED_PATH` in train.sh still uses placeholder path
- [ ] Consider adding action bounds to CNN policy config as well
