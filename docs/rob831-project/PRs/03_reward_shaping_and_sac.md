# PR: Action Smoothness Penalty + SAC Config + PPO Hyperparameter Tuning

**Source branch:** `ppo-config2`
**Target branch:** `roboeval-integration`

---

## Summary

Address poor training performance on the RoboEval LiftPot task where the bimanual arms move too fast and the robot base topples over. Three changes: (1) add an action rate penalty to the dense reward to discourage jerky movements, (2) create a tuned PPO v2 config with stabilized hyperparameters, and (3) create a SAC config as an alternative algorithm better suited for high-dimensional continuous control.

## Motivation

Training run `mlp_policy_38141312` with the original PPO config (`roboeval_liftpot_ppo_mlp.yaml`) showed:
- **Eval return stuck at ~-5 to -7** with no improvement over 200 epochs
- **Train return wildly unstable** — brief spike then collapse
- **Eval videos**: arms flail rapidly, base topples from momentum

Root causes identified:
1. **No action smoothness penalty** — reward has reach/grasp/lift/pose/success terms but nothing penalising large or jerky actions, so the policy learns to flail
2. **Relative reward (`use_rel_reward: True`)** — computing reward deltas creates high-variance gradients that destabilise PPO's advantage estimation
3. **Aggressive hyperparameters** — clip ratio 0.2, lr 3e-4, and only 8 update epochs allow large destabilising policy updates
4. **PPO limitations for 16-DOF continuous control** — on-policy PPO is sample-inefficient and struggles with high-dimensional action spaces compared to off-policy methods like SAC

## Key Changes

### 1. Action Rate Penalty in Dense Reward Wrapper

**File:** `rlinf/envs/roboeval/roboeval_env.py`

Added `w_action_rate` parameter to `_LiftPotDenseRewardWrapper`:
- Computes squared L2 norm of action change between consecutive steps: `||a_t - a_{t-1}||^2`
- Penalises large action deltas, encouraging smooth slow movements
- Default weight is 0.0 (backward compatible — existing configs unchanged)
- Added `reset()` override to clear the previous action buffer
- Reward kwargs (`w_reach`, `w_grasp`, `w_lift`, `w_pose`, `w_success`, `w_action_rate`) are now read from the env config and forwarded to the wrapper, so weights can be tuned per-experiment without code changes

### 2. Tuned PPO Config (v2)

**File:** `examples/embodiment/config/roboeval_liftpot_ppo_mlp_v2.yaml`

Changes from `roboeval_liftpot_ppo_mlp.yaml`:

| Parameter | v1 | v2 | Reason |
|-----------|----|----|--------|
| `clip_ratio_high/low` | 0.2 | 0.1 | Smaller policy updates for stability |
| `update_epoch` | 8 | 16 | Better sample usage per rollout |
| `entropy_bonus` | 0.0 | 0.01 | Encourage structured exploration |
| `use_rel_reward` | True | False | Raw dense reward is less noisy |
| `lr` / `value_lr` | 3e-4 | 1e-4 | Slower learning for stability |
| `max_epochs` | 200 | 400 | More training time |
| `w_action_rate` | (none) | 0.1 | Penalise jerky movements |

### 3. SAC Config (new algorithm)

**File:** `examples/embodiment/config/roboeval_liftpot_sac_mlp.yaml`

Modelled after `maniskill_sac_mlp.yaml`, adapted for RoboEval LiftPot:
- **Off-policy** with replay buffer (cache_size=10000 trajectories)
- **Auto-tuned entropy** (`alpha_type: softplus`, `target_entropy: -16` = -action_dim)
- **Soft target updates** (`tau: 0.005`)
- **Q-head**: dual Q-networks via `add_q_head: True`, `add_value_head: False`
- `ignore_terminations: True` for infinite-horizon bootstrapping (standard for SAC on manipulation)
- `max_steps_per_rollout_epoch: 2` (collect few transitions per epoch, train many updates — SAC is off-policy)
- `use_rel_reward: False` + `w_action_rate: 0.1` for smooth dense reward

SAC advantages for this task:
- Entropy regularisation produces structured exploration (vs PPO's undirected Gaussian noise)
- Off-policy = much more sample efficient (reuses past experience)
- Better suited for 16-DOF continuous action spaces

### 4. Random Agent Baseline Script

**File:** `scripts/eval_random_agent.py`

Standalone script to evaluate a random policy on LiftPot, providing a lower-bound baseline for learning curves:
- Instantiates a single `RoboEvalEnv` from any experiment config, overriding Hydra interpolations (`group_size`, `video_cfg`) that don't resolve outside the full Hydra config tree
- Runs N episodes (default 3) with uniform random actions sampled from the action space bounds
- Reports mean and std of episode returns, and optionally saves results to a JSON file for use by `plot_return.py`
- Uses absolute (non-relative) dense reward to match eval-time reward computation

### 5. Multi-Experiment Plotting Script

**File:** `scripts/plot_return.py`

Rewritten to support overlaying multiple experiments on a single plot:
- Accepts repeated `--config` / `--results` / `--name` flags, matched 1:1 — each experiment gets its own `steps_per_epoch` computed from its config (e.g. PPO: 32×200=6400, SAC: 32×2=64)
- **Single experiment**: plots both train and eval curves
- **Multiple experiments**: plots only eval curves to keep the comparison clean
- `--random-baseline <JSON>` or `--random-return <float>`: draws a dotted horizontal line for the random agent baseline
- `--title` overrides the auto-generated title
- `--output` accepts a directory (auto-generates filename) or a file path

### 6. SLURM Scripts

**Files:** `slurm/ppo_train.sh` (renamed from `slurm/train.sh`), `slurm/sac_train.sh` (new)

- `ppo_train.sh`: renamed for clarity, now runs the PPO v2 config by default (`roboeval_liftpot_ppo_mlp_v2`), with commented-out blocks for original PPO and CNN configs
- `sac_train.sh`: new SLURM script for SAC training with `RAY_ADDRESS=local` (required for SAC's off-policy replay), 8-hour wall time (longer than PPO due to replay buffer warmup)
- Both scripts write results to `../results/<job_name>_<job_id>/` and move log files to `LOG_DIR` after completion

## Files Changed

| File | Change |
|------|--------|
| `rlinf/envs/roboeval/roboeval_env.py` | Action rate penalty in `_LiftPotDenseRewardWrapper`; forward reward kwargs from config |
| `examples/embodiment/config/roboeval_liftpot_ppo_mlp_v2.yaml` | New tuned PPO config |
| `examples/embodiment/config/roboeval_liftpot_sac_mlp.yaml` | New SAC config |
| `scripts/eval_random_agent.py` | New random agent evaluation script with JSON output |
| `scripts/plot_return.py` | Rewritten to support multi-experiment comparison plots with random baseline |
| `slurm/ppo_train.sh` | Renamed from `train.sh`, updated to use PPO v2 config |
| `slurm/sac_train.sh` | New SLURM script for SAC training |
| `slurm/submit_job_test.sh` | Removed (superseded by per-algorithm scripts) |

## Backward Compatibility

- `_LiftPotDenseRewardWrapper` defaults `w_action_rate=0.0` — existing configs produce identical rewards
- Reward kwargs forwarding only reads keys that are present in the env config; missing keys use class defaults
- Original `roboeval_liftpot_ppo_mlp.yaml` is unchanged

## Testing Plan

- [ ] Run PPO v2 config and compare learning curve against `mlp_policy_38141312` baseline
- [ ] Run SAC config and verify replay buffer fills and Q-values converge
- [ ] Check eval videos for both: arms should move smoothly, base should not topple
- [ ] Verify original PPO config still produces identical results (backward compat)
- [ ] Ablation: try `w_action_rate` values 0.05, 0.1, 0.2 to find best smoothness/progress trade-off
- [ ] Run `eval_random_agent.py` to establish random baseline, then use `plot_return.py` to generate PPO vs SAC comparison plot

## Recommended Run Order

1. **Random baseline** — run `eval_random_agent.py` to get the lower-bound return
2. **PPO v2 first** — fastest to validate since it's a direct comparison to the v1 baseline
3. **SAC second** — takes longer to start learning (replay buffer warmup) but should ultimately achieve better performance
4. **Comparison plot** — use `plot_return.py` with both results dirs and the random baseline JSON
5. If SAC works well, try `q_head_type: crossq` for faster Q-network training (no target network lag)
