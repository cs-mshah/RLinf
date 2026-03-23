# PR: Action Smoothness Penalty + SAC Config + PPO Hyperparameter Tuning

**Source branch:** `reward-shaping-sac`
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

## Files Changed

| File | Change |
|------|--------|
| `rlinf/envs/roboeval/roboeval_env.py` | Action rate penalty in `_LiftPotDenseRewardWrapper`; forward reward kwargs from config |
| `examples/embodiment/config/roboeval_liftpot_ppo_mlp_v2.yaml` | New tuned PPO config |
| `examples/embodiment/config/roboeval_liftpot_sac_mlp.yaml` | New SAC config |

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

## Recommended Run Order

1. **PPO v2 first** — fastest to validate since it's a direct comparison to the v1 baseline
2. **SAC second** — takes longer to start learning (replay buffer warmup) but should ultimately achieve better performance
3. If SAC works well, try `q_head_type: crossq` for faster Q-network training (no target network lag)
