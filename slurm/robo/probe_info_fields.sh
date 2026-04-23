#!/bin/bash
#SBATCH --job-name=probe_info_fields
#SBATCH --partition=ROBO
#SBATCH --reservation=ROBOcis220039p
#SBATCH --nodelist=robo-gh005
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:10:00
#SBATCH -A cis220039p
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Probes which of the SE(3) wrapper's expected info fields are actually
# populated by RoboTwin's VectorEnv after a reset() + step() cycle on lift_pot.

source "${SLURM_SUBMIT_DIR}/slurm/robo/_common.sh"
set +eo pipefail

python - <<'PY'
import os, numpy as np, traceback
from omegaconf import OmegaConf

cfg = OmegaConf.load(os.path.join(os.environ["REPO_PATH"],
    "examples/embodiment/config/env/robotwin_lift_pot.yaml"))
cfg.total_num_envs = 1
cfg.group_size = 1
cfg.assets_path = os.environ["ROBOTWIN_ASSETS_PATH"]
cfg.seeds_path = os.path.join(os.environ["REPO_PATH"],
    "rlinf/envs/robotwin/seeds/eval_seeds.json")
cfg.video_cfg.save_video = False
cfg.enable_offload = False

from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
env = RoboTwinEnv(cfg=cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None)

expected = [
    "pot_pos", "pot_rot_mat",
    "target_pos", "target_rot_mat",
    "left_gripper_pose", "right_gripper_pose",
    "left_handle_pose", "right_handle_pose",
    "grasp_left_success", "grasp_right_success",
    "lift_distance", "success",
]

print("\n=== reset ===")
obs, info = env.reset()
print("  info type:", type(info).__name__)
print("  info keys:", sorted(list(info.keys()) if isinstance(info, dict) else []))

# Peek underlying venv.envs[0] to see what its own step info has.
try:
    raw = env.venv.envs[0]
    print("  underlying env attrs with pot/handle/gripper:")
    for n in dir(raw):
        if any(k in n.lower() for k in ["pot", "handle", "gripper", "target"]):
            print("    .", n)
except Exception as e:
    print("  underlying env introspection failed:", e)

print("\n=== step ===")
import numpy as np
a = np.zeros((1, 1, 14), dtype=np.float32)  # [n_envs, horizon, action_dim]
obs, rew, term, trunc, info = env.step(a)
if isinstance(info, dict):
    keys = sorted(info.keys())
    print("  info keys:", keys)
else:
    print("  info type:", type(info).__name__)
    # Could be list-of-dicts collated into dict-of-lists.
    if isinstance(info, dict):
        keys = list(info.keys())
    else:
        keys = []

print("\n=== expected SE(3) fields availability ===")
flat_info = info if isinstance(info, dict) else {}
for k in expected:
    present = k in flat_info
    mark = "OK" if present else "MISSING"
    extra = ""
    if present:
        v = flat_info[k]
        if hasattr(v, "shape"):
            extra = f" shape={tuple(v.shape)}"
        else:
            extra = f" type={type(v).__name__}"
    print(f"  [{mark}] {k}{extra}")
PY
