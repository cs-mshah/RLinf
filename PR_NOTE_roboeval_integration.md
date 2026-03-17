# PR: RoboEval LiftPot Environment Integration

**Source branch:** `roboeval-env`
**Target branch:** `roboeval-integration`

---

## Summary

Integrate the RoboEval LiftPot environment into the RLinf embodied RL training framework, enabling PPO-based training on the LiftPot manipulation task.

## Changes

### New files
- `rlinf/envs/roboeval/` — RoboEval environment wrapper package
  - `__init__.py`
  - `roboeval_env.py` — `RoboEvalEnv` class that wraps RoboEval's `LiftPot` task into RLinf's env interface, including dense reward shaping (inlined from `rob16831/roboeval_wrapper.py`)
- `examples/embodiment/config/roboeval_liftpot_ppo_mlp.yaml` — Hydra config for PPO + MLP training on LiftPot
- `examples/embodiment/config/env/roboeval_liftpot_state.yaml` — Environment-specific config (obs dims, action space, num_envs)

### Modified files
- `rlinf/envs/__init__.py` — Register `RoboEvalEnv` in the env registry
- `rlinf/envs/action_utils.py` — Add action space utilities for RoboEval's continuous action space

## Prerequisites
- RoboEval package installed (`pip install -e /path/to/RoboEval`)
- MuJoCo Menagerie submodule initialized in RoboEval (`git submodule update --init --recursive`)
- `MUJOCO_GL=egl` for headless environments (set in run script)
- `EMBODIED_PATH` env var pointing to the embodied config directory

## Testing done
- Environment initializes and loads MuJoCo models successfully (MuJoCo 3.3.3, EGL backend)
- Ray cluster spawns env workers with correct placement on V100 GPU
- Still tuning `total_num_envs` to avoid OOM (128 may be too high for single-node)

## Known issues / TODOs
- [ ] Reduce default `total_num_envs` (128 train envs may OOM on single V100-32GB node; try 16-32)
- [ ] Verify full training loop completes end-to-end
- [ ] Add eval video logging support
- [ ] Test multi-node scaling
- [ ] Confirm reward shaping produces learning signal (check TensorBoard curves)
