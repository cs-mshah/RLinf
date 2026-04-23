#!/bin/bash
#SBATCH --job-name=diag_vla_unnorm
#SBATCH --partition=ROBO
#SBATCH --reservation=ROBOcis220039p
#SBATCH --nodelist=robo-gh005
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:15:00
#SBATCH -A cis220039p
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Probe how RLinf's openvla_oft loader populates norm_stats/unnorm_key for our
# SFT checkpoint, and print the ROBOT_PLATFORM preset (ALOHA vs PIPER mismatch
# hypothesis H1).

source "${SLURM_SUBMIT_DIR}/slurm/robo/_common.sh"
set +eo pipefail  # don't let probe failures kill the diag

python - <<'PY'
import os, sys, json, traceback

print("=" * 60)
print("env: ROBOT_PLATFORM=", os.environ.get("ROBOT_PLATFORM"))
print("env: VLA_CKPT_PATH=", os.environ.get("VLA_CKPT_PATH"))
print("=" * 60)

# 1. Load dataset_statistics.json directly.
try:
    with open(os.path.join(os.environ["VLA_CKPT_PATH"], "dataset_statistics.json")) as f:
        ds = json.load(f)
    print("dataset_statistics.json keys:", list(ds.keys()))
except Exception as e:
    print("cannot read dataset_statistics.json:", e)

# 2. Inspect prismatic's constants for ALOHA vs any PIPER preset.
try:
    from prismatic.vla import constants as C
    # Constants module sets globals based on ROBOT_PLATFORM env.
    print("\n--- prismatic.vla.constants loaded with ROBOT_PLATFORM=", os.environ.get("ROBOT_PLATFORM"))
    for name in dir(C):
        if name.isupper() and not name.startswith("_"):
            val = getattr(C, name)
            if not callable(val):
                print(f"  C.{name} = {val}")
except Exception as e:
    print("prismatic constants import failed:", e)
    traceback.print_exc()

# 3. Exercise the official RLinf loader for the checkpoint.
try:
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "model_path": os.environ["VLA_CKPT_PATH"],
        "implement_version": "official",
        "model_type": "openvla_oft",
        "action_dim": 14,
        "use_proprio": True,
        "proprio_dim": 14,
        "use_film": False,
        "num_action_chunks": 25,
        "unnorm_key": "lift_pot_1k",
        "center_crop": True,
        "is_lora": True,
        "precision": "bf16",
        "value_type": "action_level",
        "max_prompt_length": 512,
    })
    from rlinf.models.embodiment.openvla_oft.official import __init__ as official_init
    print("\n--- loader module:", official_init.__file__)
    # Try the top-level load function.
    import rlinf.models.embodiment.openvla_oft.official as official
    funcs = [n for n in dir(official) if "load" in n.lower() or "build" in n.lower() or "make" in n.lower()]
    print("  candidate loader funcs:", funcs)
except Exception as e:
    print("RLinf loader probe failed:", e)
    traceback.print_exc()

# 4. Inspect how ROBOT_PLATFORM is consumed.
try:
    import prismatic.vla.constants as C2
    platform = getattr(C2, "ROBOT_PLATFORM", None)
    action_dim = getattr(C2, "ACTION_DIM", None)
    proprio_dim = getattr(C2, "PROPRIO_DIM", None)
    n_chunks = getattr(C2, "NUM_ACTIONS_CHUNK", None)
    print("\nACTIVE PLATFORM:", platform)
    print("  ACTION_DIM=", action_dim, "PROPRIO_DIM=", proprio_dim, "NUM_ACTIONS_CHUNK=", n_chunks)
except Exception as e:
    print("constant-inspection failed:", e)
PY
