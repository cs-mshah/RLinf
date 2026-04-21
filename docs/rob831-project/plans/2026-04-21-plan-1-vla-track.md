# Plan 1 — VLA Track (Stages 0–3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** End-to-end VLA pipeline on RoboTwin `lift_pot` — eval the pretrained SFT checkpoint (B4), RL-fine-tune it on one GPU (M1), and run the same fine-tune with an SE(3)/SO(3)-based reward (M2b).

**Architecture:** Reuses the existing `robotwin_lift_pot_grpo_openvlaoft.yaml` training pipeline. Adds (a) a 1-GPU scaled-down config override, (b) a new `LiftPotSE3RewardWrapper` env wrapper with companion Lie-group math utilities, (c) SLURM scripts targeting the `ROBO` reservation. No new training backends, no new model code — all work is config + env-side plumbing.

**Tech Stack:** PyTorch / FSDP / Hydra / Gymnasium / RoboTwin / HuggingFace transformers (`openvla-oft`) / PSC Bridges-2 SLURM + apptainer.

**Spec:** `docs/rob831-project/specs/2026-04-21-robotwin-liftpot-vla-rl-design.md`

---

## File Structure

Files created or modified by this plan:

**New files (code):**
- `rlinf/envs/robotwin/se3_math.py` — SO(3) geodesic distance + SE(3) log-map utilities (pure PyTorch/NumPy, no new deps).
- `rlinf/envs/robotwin/lift_pot_reward_wrapper.py` — `LiftPotSE3RewardWrapper` gymnasium wrapper.
- `tests/unit_tests/test_se3_math.py` — unit tests for the Lie-group utilities.
- `tests/unit_tests/test_lift_pot_reward_wrapper.py` — unit tests for the reward wrapper using a mock env.

**New files (configs):**
- `examples/embodiment/config/robotwin_lift_pot_grpo_openvlaoft_eval_1gpu.yaml` — B4 eval-only config, 1 GPU.
- `examples/embodiment/config/robotwin_lift_pot_grpo_openvlaoft_1gpu.yaml` — M1 training config, 1 GPU, scaled down.
- `examples/embodiment/config/robotwin_lift_pot_grpo_openvlaoft_1gpu_se3.yaml` — M2b: M1 + SE(3) reward.
- `examples/embodiment/config/env/robotwin_lift_pot_se3.yaml` — env config variant that wraps with the SE(3) reward.

**New files (SLURM):**
- `slurm/robo/_common.sh` — shared env setup sourced by all robo SLURM scripts.
- `slurm/robo/b4_eval.sh` — Stage 1.
- `slurm/robo/m1_vla_grpo.sh` — Stage 2.
- `slurm/robo/m2b_vla_grpo_se3.sh` — Stage 3.

**Modified files:**
- `rlinf/envs/robotwin/robotwin_env.py` — minimal change: if `cfg.reward_variant == "se3"`, attach `LiftPotSE3RewardWrapper` and expose pot/handle/target poses in `infos`.

**Plan boundary:** This plan ends after M2b produces eval numbers. State-obs mode for the RoboTwin env, MLP baselines (B1/B2/M2a), and MBPO (B3) are in Plan 2 — they share the same env file but a *different* modification to it (add state-obs mode), so deferring until we have M1/M2b results is a natural decomposition.

---

## Pre-task conventions

- Working directory for all shell commands: `/ocean/projects/cis220039p/hkwon/projects/RLinf`.
- Python interpreter: the `rlinf-openvlaoft` conda env built in Task 2.
- Commit messages: conventional-commit-style, present tense, one PR per logical task — matches the existing `git log` style in this repo.
- Reservation node for all submissions: `robo-gh005` (reservation `ROBOcis220039p`). Confirm in Task 1 before first submit.
- Whenever a task submits a training job, the user monitors via `squeue -u $USER` and the job's `.out` / `.err` log in `slurm/logs/`.

---

## Task 1: Preflight — verify reservation and repo state

**Files:** none (read-only checks).

- [ ] **Step 1: Confirm SLURM reservation is active and partition name is correct**

Run:
```bash
sinfo -p ROBO -o "%N %t %G %C"
scontrol show reservation ROBOcis220039p | head -5
```

Expected:
- `robo-gh005 ... gpu:h100:8 ...` is in `ROBO` partition and is `resv` or `idle+resv`.
- Reservation `State=ACTIVE`, accounts includes `cis220039p`, end time ≥ 2026-04-28.

If partition is not called `ROBO`, record the actual name and replace it in every SLURM script in Task 5+.

- [ ] **Step 2: Confirm git working tree is clean and on the correct branch**

Run:
```bash
git status
git branch --show-current
```

Expected: `On branch roboeval-integration`, working tree clean (or, at worst, the in-progress spec edits from the brainstorming session). Stash/commit anything stray before proceeding.

- [ ] **Step 3: Verify the existing RoboTwin GRPO config still parses**

Run:
```bash
python -c "from omegaconf import OmegaConf; c = OmegaConf.load('examples/embodiment/config/robotwin_lift_pot_grpo_openvlaoft.yaml'); print(c.actor.model.model_path)"
```

Expected: prints `/path/to/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot` — the placeholder we'll replace in Task 3.

- [ ] **Step 4: No commit (read-only task).**

---

## Task 2: Preflight — build `rlinf-openvlaoft` conda env

**Files:** none (creates env outside repo). Record any hand-edits to requirements in a new note `docs/rob831-project/env_notes.md` if install.sh fails.

- [ ] **Step 1: Load system modules and create the conda env**

Run:
```bash
module load anaconda3 cuda
conda create -n rlinf-openvlaoft python=3.11 -y
source activate rlinf-openvlaoft
```

Expected: `which python` points to `~/.conda/envs/rlinf-openvlaoft/bin/python`.

- [ ] **Step 2: Run RLinf's install script for openvla-oft + robotwin**

Run:
```bash
bash requirements/install.sh embodied --model openvla-oft --env robotwin
```

Expected: script completes without error. On failure, capture the error, read `requirements/install.sh` to find which step failed, and fall back to manual pip installs. Record any deviations in `docs/rob831-project/env_notes.md`.

- [ ] **Step 3: Verify RLinf itself is importable from this env**

Run:
```bash
python -c "import rlinf; print(rlinf.__file__)"
```

Expected: prints a path under the current repo (editable install) or the conda env's site-packages. If `ImportError`, run `pip install -e .` from the repo root.

- [ ] **Step 4: Verify `openvla_oft` model code is importable**

Run:
```bash
python -c "from rlinf.models.embodiment.openvla_oft import __init__; print('ok')"
```

Expected: `ok`.

- [ ] **Step 5: Verify `robotwin` task package is importable**

Run:
```bash
python -c "import robotwin; print(robotwin.__file__)"
```

Expected: prints a valid path. If this fails, the RoboTwin repo must be cloned and installed per `requirements/envs/robotwin*` — record the resolution in `env_notes.md`.

- [ ] **Step 6: Commit env notes if created**

```bash
git add docs/rob831-project/env_notes.md 2>/dev/null || true
git commit -m "docs(rob831): env install notes for rlinf-openvlaoft" || true
```

---

## Task 3: Preflight — download SFT checkpoint + smoke test

**Files:** none checked in (the checkpoint lives under `$HF_HOME`).

- [ ] **Step 1: Download the SFT checkpoint**

Run:
```bash
huggingface-cli download RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot \
  --local-dir $HOME/OCEANDIR/checkpoints/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot \
  --local-dir-use-symlinks False
```

Expected: progress bars finish with no errors. Checkpoint directory contains `adapter_model.safetensors` or `*.safetensors` weights plus `config.json`. Size should be on the order of a few GB.

If rate-limited or auth-required, run `huggingface-cli login` with your HF token first.

- [ ] **Step 2: Smoke test — load the checkpoint via RLinf's model loader**

Create and run a single-use script `/tmp/smoke_load.py`:
```python
import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/OCEANDIR/.cache/huggingface"))
from omegaconf import OmegaConf
cfg = OmegaConf.load("examples/embodiment/config/model/openvla_oft.yaml")
cfg.model_path = os.path.expanduser("~/OCEANDIR/checkpoints/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot")
cfg.implement_version = "official"
cfg.action_dim = 14
cfg.use_proprio = True
cfg.proprio_dim = 14
cfg.num_action_chunks = 25
cfg.unnorm_key = "lift_pot_1k"
cfg.is_lora = True
print("config OK:", cfg.model_path)
```

Run:
```bash
python /tmp/smoke_load.py
```

Expected: `config OK: ...` with the full local path. This validates that the config system accepts the RoboTwin-specific fields.

- [ ] **Step 3: No commit (checkpoint lives outside repo).**

---

## Task 4: Stage 1 — B4 eval config

**Files:**
- Create: `examples/embodiment/config/robotwin_lift_pot_grpo_openvlaoft_eval_1gpu.yaml`

- [ ] **Step 1: Create the eval-only 1-GPU config**

Write the file:
```yaml
# B4: Eval-only VLA (OpenVLA-OFT SFT'd on RoboTwin lift_pot) on 1 H100.
# Derived from robotwin_lift_pot_grpo_openvlaoft.yaml — overrides:
#   - runner.only_eval = True  (skips training loop)
#   - cluster placement = single GPU
#   - actor.model.model_path = local checkpoint directory
#   - asset paths → real values on PSC

defaults:
  - robotwin_lift_pot_grpo_openvlaoft
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null

cluster:
  num_nodes: 1
  component_placement:
    actor, env, rollout: 0            # single GPU

runner:
  only_eval: True
  logger:
    experiment_name: "b4_vla_zeroshot"
    logger_backends: ["tensorboard"]

env:
  train:
    total_num_envs: 16                # small — eval only
    assets_path: ${oc.env:ROBOTWIN_ASSETS_PATH}
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
  eval:
    total_num_envs: 16
    assets_path: ${oc.env:ROBOTWIN_ASSETS_PATH}
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    video_cfg:
      save_video: True
      video_base_dir: ${runner.logger.log_path}/video/eval

actor:
  # override placeholder path — the actual downloaded checkpoint
  model:
    model_path: ${oc.env:VLA_CKPT_PATH}
```

- [ ] **Step 2: Verify the override chain parses**

Run:
```bash
EMBODIED_PATH=examples/embodiment/ \
REPO_PATH=$PWD \
ROBOTWIN_ASSETS_PATH=/tmp/_fake_assets \
VLA_CKPT_PATH=/tmp/_fake_ckpt \
python -c "
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
with initialize_config_dir(version_base='1.1', config_dir='$PWD/examples/embodiment/config'):
    cfg = compose(config_name='robotwin_lift_pot_grpo_openvlaoft_eval_1gpu')
print('only_eval:', cfg.runner.only_eval)
print('num_envs eval:', cfg.env.eval.total_num_envs)
"
```

Expected: `only_eval: True`, `num_envs eval: 16`. If hydra errors, fix the override syntax before moving on.

- [ ] **Step 3: Commit**

```bash
git add examples/embodiment/config/robotwin_lift_pot_grpo_openvlaoft_eval_1gpu.yaml
git commit -m "feat(config): add B4 eval-only 1-GPU VLA config for RoboTwin lift_pot"
```

---

## Task 5: Stage 1 — B4 SLURM scripts

**Files:**
- Create: `slurm/robo/_common.sh`
- Create: `slurm/robo/b4_eval.sh`

- [ ] **Step 1: Create the shared SLURM helper**

```bash
# slurm/robo/_common.sh
# Sourced by each robo SLURM script. Defines common paths and env vars.

set -eo pipefail

module load anaconda3 cuda
source activate rlinf-openvlaoft

export REPO_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment/"
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH}"

# RoboTwin assets — set this to the real path in your own env or .bashrc
export ROBOTWIN_ASSETS_PATH=${ROBOTWIN_ASSETS_PATH:-$HOME/OCEANDIR/data/robotwin_assets}
export VLA_CKPT_PATH=${VLA_CKPT_PATH:-$HOME/OCEANDIR/checkpoints/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot}

export EXPERIMENT_NAME="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
export RESULTS_DIR="${REPO_PATH}/../results/${EXPERIMENT_NAME}"
mkdir -p "${RESULTS_DIR}"

echo "[_common] REPO_PATH=${REPO_PATH}"
echo "[_common] VLA_CKPT_PATH=${VLA_CKPT_PATH}"
echo "[_common] ROBOTWIN_ASSETS_PATH=${ROBOTWIN_ASSETS_PATH}"
echo "[_common] RESULTS_DIR=${RESULTS_DIR}"
```

- [ ] **Step 2: Create the B4 SLURM script**

```bash
# slurm/robo/b4_eval.sh
#!/bin/bash
#SBATCH --job-name=b4_vla_zeroshot
#SBATCH --partition=ROBO
#SBATCH --reservation=ROBOcis220039p
#SBATCH --nodelist=robo-gh005
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH -A cis220039p
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

cd "${REPO_PATH}"
python examples/embodiment/eval_embodied_agent.py \
  --config-path=examples/embodiment/config/ \
  --config-name=robotwin_lift_pot_grpo_openvlaoft_eval_1gpu \
  runner.logger.log_path="${RESULTS_DIR}"

echo "[done] B4 eval completed. Results in ${RESULTS_DIR}."
```

Make it executable:
```bash
mkdir -p slurm/robo slurm/logs
chmod +x slurm/robo/b4_eval.sh slurm/robo/_common.sh
```

- [ ] **Step 3: Dry-run the SLURM script (syntax check)**

Run:
```bash
sbatch --test-only slurm/robo/b4_eval.sh
```

Expected: `sbatch: Job ... is valid for submission` (no actual submission). If it complains about partition/reservation, revise against the Task 1 findings.

- [ ] **Step 4: Commit**

```bash
git add slurm/robo/_common.sh slurm/robo/b4_eval.sh
git commit -m "feat(slurm): add B4 VLA-zero-shot eval SLURM scripts for ROBO reservation"
```

---

## Task 6: Stage 1 — Submit B4 and verify the result

**Files:** none.

- [ ] **Step 1: Submit B4**

Run:
```bash
sbatch slurm/robo/b4_eval.sh
```

Expected: prints `Submitted batch job <JOB_ID>`. Record the job ID.

- [ ] **Step 2: Monitor the job**

Run:
```bash
squeue -u $USER
# and, once it starts:
tail -f slurm/logs/b4_vla_zeroshot_<JOB_ID>.out
```

Expected: job transitions `PENDING → RUNNING → COMPLETED` in ~30–60 min. Log shows env init, then per-step eval progress, then a final summary with a success rate number.

- [ ] **Step 3: Extract the B4 success rate from TensorBoard**

Run:
```bash
python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, os, sys
acc = EventAccumulator(glob.glob(f'{os.path.expanduser(\"~\")}/OCEANDIR/projects/RLinf/../results/b4_vla_zeroshot_*/tensorboard')[-1])
acc.Reload()
# success_once is the primary metric
for tag in acc.Tags()['scalars']:
    if 'success' in tag:
        evs = acc.Scalars(tag)
        print(tag, '=', evs[-1].value if evs else 'n/a')
"
```

Expected: prints `eval/success_once = <float>` among other keys. Write the value down.

- [ ] **Step 4: Decision gate**

- If `success_once < 0.20`: STOP. Open a scratch note at `docs/rob831-project/notes/b4_debug_<date>.md`, log current symptoms, and check one-by-one: `unnorm_key` set to `lift_pot_1k`?, `implement_version="official"`?, cameras matching SFT distribution (head + 2 wrists, 224×224 or 128×128)?, correct LoRA adapter loaded? Do not proceed until B4 > 20%.
- If `success_once >= 0.20`: proceed to Task 7.

- [ ] **Step 5: Commit a summary note**

```bash
mkdir -p docs/rob831-project/results
cat > docs/rob831-project/results/b4_zeroshot.md <<EOF
# B4 — VLA Zero-Shot Results

- Job: \`b4_vla_zeroshot_<JOB_ID>\`
- Checkpoint: \`RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot\`
- Eval envs: 16, horizon 200
- **success_once: X.XX**
- Eval videos: \`results/b4_vla_zeroshot_<JOB_ID>/video/eval/\`

Decision gate: passed / failed.
EOF
git add docs/rob831-project/results/b4_zeroshot.md
git commit -m "docs(rob831): log B4 zero-shot results"
```

---

## Task 7: Stage 2 — M1 scaled-down VLA-GRPO config

**Files:**
- Create: `examples/embodiment/config/robotwin_lift_pot_grpo_openvlaoft_1gpu.yaml`

- [ ] **Step 1: Write the M1 training config**

```yaml
# M1: RL fine-tune of VLA (OpenVLA-OFT) on RoboTwin lift_pot, 1 H100, LoRA.
# Derived from robotwin_lift_pot_grpo_openvlaoft.yaml with the rescaling in
# docs/rob831-project/specs/2026-04-21-robotwin-liftpot-vla-rl-design.md §4.5.

defaults:
  - robotwin_lift_pot_grpo_openvlaoft
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null

cluster:
  num_nodes: 1
  component_placement:
    actor, env, rollout: 0

runner:
  only_eval: False
  max_epochs: 150
  val_check_interval: 10
  save_interval: 10
  logger:
    experiment_name: "m1_vla_grpo_1gpu"
    logger_backends: ["tensorboard"]

algorithm:
  group_size: 4                       # rescaled from 8

env:
  train:
    total_num_envs: 32                # per-group (8) × num_group (4) matches group_size=4 with 4 groups
    group_size: ${algorithm.group_size}
    assets_path: ${oc.env:ROBOTWIN_ASSETS_PATH}
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
  eval:
    total_num_envs: 16
    assets_path: ${oc.env:ROBOTWIN_ASSETS_PATH}
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    video_cfg:
      save_video: True

actor:
  micro_batch_size: 2                 # fits H100 80 GB with LoRA + grad checkpointing
  global_batch_size: 32               # = micro_batch × grad_accum = 2 × 16
  model:
    model_path: ${oc.env:VLA_CKPT_PATH}
  fsdp_config:
    strategy: "no_shard"              # single-GPU, no FSDP sharding
    gradient_checkpointing: True
```

- [ ] **Step 2: Verify the override chain parses**

Run:
```bash
EMBODIED_PATH=examples/embodiment/ \
REPO_PATH=$PWD \
ROBOTWIN_ASSETS_PATH=/tmp/_fake_assets \
VLA_CKPT_PATH=/tmp/_fake_ckpt \
python -c "
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
with initialize_config_dir(version_base='1.1', config_dir='$PWD/examples/embodiment/config'):
    cfg = compose(config_name='robotwin_lift_pot_grpo_openvlaoft_1gpu')
print('max_epochs:', cfg.runner.max_epochs)
print('micro_batch:', cfg.actor.micro_batch_size)
print('global_batch:', cfg.actor.global_batch_size)
print('group_size:', cfg.algorithm.group_size)
"
```

Expected: `max_epochs: 150`, `micro_batch: 2`, `global_batch: 32`, `group_size: 4`.

- [ ] **Step 3: Commit**

```bash
git add examples/embodiment/config/robotwin_lift_pot_grpo_openvlaoft_1gpu.yaml
git commit -m "feat(config): add 1-GPU scaled M1 VLA-GRPO config"
```

---

## Task 8: Stage 2 — M1 SLURM script

**Files:**
- Create: `slurm/robo/m1_vla_grpo.sh`

- [ ] **Step 1: Write the M1 SLURM script**

```bash
# slurm/robo/m1_vla_grpo.sh
#!/bin/bash
#SBATCH --job-name=m1_vla_grpo
#SBATCH --partition=ROBO
#SBATCH --reservation=ROBOcis220039p
#SBATCH --nodelist=robo-gh005
#SBATCH --gres=gpu:h100:1
#SBATCH --time=16:00:00
#SBATCH -A cis220039p
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

cd "${REPO_PATH}"
python examples/embodiment/train_embodied_agent.py \
  --config-path=examples/embodiment/config/ \
  --config-name=robotwin_lift_pot_grpo_openvlaoft_1gpu \
  runner.logger.log_path="${RESULTS_DIR}"

echo "[done] M1 training completed. Results in ${RESULTS_DIR}."
```

Make it executable:
```bash
chmod +x slurm/robo/m1_vla_grpo.sh
```

- [ ] **Step 2: Dry-run**

Run:
```bash
sbatch --test-only slurm/robo/m1_vla_grpo.sh
```

Expected: `Job ... is valid`.

- [ ] **Step 3: Commit**

```bash
git add slurm/robo/m1_vla_grpo.sh
git commit -m "feat(slurm): add M1 VLA-GRPO 1-GPU training SLURM script"
```

---

## Task 9: Stage 2 — Submit M1

**Files:** none.

- [ ] **Step 1: Submit M1**

Run:
```bash
sbatch slurm/robo/m1_vla_grpo.sh
```

Expected: `Submitted batch job <JOB_ID>`. Record the ID.

- [ ] **Step 2: Verify it started running within 5 minutes**

Run:
```bash
squeue -u $USER -j <JOB_ID>
```

Expected: `ST` column shows `R` (running). If `PD` (pending) persists for >5 min on an empty reservation, something is wrong — check `squeue -u $USER -j <JOB_ID> --start` for estimated start time and `scontrol show job <JOB_ID>` for a reason.

- [ ] **Step 3: Let it run while we work on Tasks 10–13 in parallel.**

The next four tasks build the SE(3) reward wrapper, which is independent of M1's training. When M1 finishes (or crosses the decision gate at epoch ~45), Task 14 evaluates convergence.

---

## Task 10: SO(3) geodesic distance utility (TDD)

**Files:**
- Create: `rlinf/envs/robotwin/se3_math.py`
- Create: `tests/unit_tests/test_se3_math.py`

- [ ] **Step 1: Write the failing test**

`tests/unit_tests/test_se3_math.py`:
```python
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rlinf.envs.robotwin.se3_math import so3_geodesic_distance


def test_so3_geodesic_identity_is_zero():
    R = np.eye(3)
    assert so3_geodesic_distance(R, R) == pytest.approx(0.0, abs=1e-7)


def test_so3_geodesic_90_deg_z_rotation():
    R_target = np.eye(3)
    R = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    assert so3_geodesic_distance(R, R_target) == pytest.approx(np.pi / 2, abs=1e-6)


def test_so3_geodesic_180_deg_is_pi():
    R_target = np.eye(3)
    R = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    assert so3_geodesic_distance(R, R_target) == pytest.approx(np.pi, abs=1e-6)


def test_so3_geodesic_is_symmetric():
    rng = np.random.default_rng(0)
    for _ in range(5):
        R1 = Rotation.random(random_state=rng).as_matrix()
        R2 = Rotation.random(random_state=rng).as_matrix()
        d1 = so3_geodesic_distance(R1, R2)
        d2 = so3_geodesic_distance(R2, R1)
        assert d1 == pytest.approx(d2, abs=1e-6)


def test_so3_geodesic_handles_numerical_edge_cases():
    """trace can exceed 3 or go below -1 due to floating-point; arccos must stay valid."""
    R = np.array([[1.0, 1e-10, 0], [-1e-10, 1.0, 0], [0, 0, 1.0]])
    d = so3_geodesic_distance(R, np.eye(3))
    assert np.isfinite(d)
    assert d >= 0.0
```

- [ ] **Step 2: Run the test — it should fail with ImportError**

Run:
```bash
pytest tests/unit_tests/test_se3_math.py -v
```

Expected: `ModuleNotFoundError: No module named 'rlinf.envs.robotwin.se3_math'`.

- [ ] **Step 3: Implement `so3_geodesic_distance`**

`rlinf/envs/robotwin/se3_math.py`:
```python
"""Lie-group utilities for the LiftPot SE(3)/SO(3) reward wrapper."""

from __future__ import annotations

import numpy as np

__all__ = ["so3_geodesic_distance", "se3_log_map"]


def so3_geodesic_distance(R: np.ndarray, R_target: np.ndarray) -> float:
    """Geodesic distance on SO(3): angle of R^T R_target.

    Returns a scalar in [0, pi]. Clamps the trace to [-1, 3] to handle
    floating-point drift in orthogonal matrices.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    R_target = np.asarray(R_target, dtype=np.float64).reshape(3, 3)
    trace = np.trace(R.T @ R_target)
    # cos(theta) = (trace - 1) / 2; clamp for numerical safety.
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cos_theta))
```

- [ ] **Step 4: Run the tests — should now pass**

Run:
```bash
pytest tests/unit_tests/test_se3_math.py::test_so3_geodesic_identity_is_zero \
       tests/unit_tests/test_se3_math.py::test_so3_geodesic_90_deg_z_rotation \
       tests/unit_tests/test_se3_math.py::test_so3_geodesic_180_deg_is_pi \
       tests/unit_tests/test_se3_math.py::test_so3_geodesic_is_symmetric \
       tests/unit_tests/test_se3_math.py::test_so3_geodesic_handles_numerical_edge_cases -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add rlinf/envs/robotwin/se3_math.py tests/unit_tests/test_se3_math.py
git commit -m "feat(robotwin): add SO(3) geodesic distance utility with tests"
```

---

## Task 11: SE(3) log-map utility (TDD)

**Files:**
- Modify: `rlinf/envs/robotwin/se3_math.py`
- Modify: `tests/unit_tests/test_se3_math.py`

- [ ] **Step 1: Add failing tests for `se3_log_map`**

Append to `tests/unit_tests/test_se3_math.py`:
```python
from rlinf.envs.robotwin.se3_math import se3_log_map


def _make_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def test_se3_log_identity_is_zero():
    T = np.eye(4)
    xi = se3_log_map(T)
    assert xi.shape == (6,)
    assert np.allclose(xi, 0.0, atol=1e-7)


def test_se3_log_pure_translation():
    T = _make_se3(np.eye(3), np.array([0.1, 0.2, -0.3]))
    xi = se3_log_map(T)
    # For R=I, v component equals t.
    assert np.allclose(xi[:3], [0.1, 0.2, -0.3], atol=1e-6)
    assert np.allclose(xi[3:], 0.0, atol=1e-7)


def test_se3_log_pure_rotation_norm_equals_angle():
    R = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    T = _make_se3(R, np.zeros(3))
    xi = se3_log_map(T)
    # omega norm equals rotation angle.
    assert np.linalg.norm(xi[3:]) == pytest.approx(np.pi / 6, abs=1e-6)
```

- [ ] **Step 2: Run — should fail with ImportError on `se3_log_map`**

Run:
```bash
pytest tests/unit_tests/test_se3_math.py -v
```

Expected: `ImportError: cannot import name 'se3_log_map'`.

- [ ] **Step 3: Implement `se3_log_map`**

Append to `rlinf/envs/robotwin/se3_math.py`:
```python
def se3_log_map(T: np.ndarray) -> np.ndarray:
    """SE(3) log map: 4x4 homogeneous transform -> 6-vector (v, omega).

    Returns [vx, vy, vz, wx, wy, wz] where (wx, wy, wz) is the axis-angle
    rotation vector and (vx, vy, vz) is the translational component in the
    tangent space. Follows Chirikjian & Kyatkin, Ch. 2 (closed-form SE(3)
    log); robust to theta -> 0 via Taylor expansion.
    """
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]

    # Rotation angle via SO(3) log.
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < 1e-8:
        # R ≈ I: omega ≈ 0, v ≈ t.
        omega = np.zeros(3)
        v = t.copy()
    else:
        # Skew-symmetric part of R gives omega-hat.
        omega_hat = (R - R.T) * (theta / (2.0 * np.sin(theta)))
        omega = np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])

        # V^{-1} matrix to map t -> v component.
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / (theta * theta)
        V_inv = (
            np.eye(3)
            - 0.5 * omega_hat
            + ((1.0 - A / (2.0 * B)) / (theta * theta)) * (omega_hat @ omega_hat)
        )
        v = V_inv @ t

    return np.concatenate([v, omega])
```

- [ ] **Step 4: Run the new tests — should pass**

Run:
```bash
pytest tests/unit_tests/test_se3_math.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add rlinf/envs/robotwin/se3_math.py tests/unit_tests/test_se3_math.py
git commit -m "feat(robotwin): add SE(3) log-map utility with tests"
```

---

## Task 12: LiftPot SE(3) reward wrapper (TDD)

**Files:**
- Create: `rlinf/envs/robotwin/lift_pot_reward_wrapper.py`
- Create: `tests/unit_tests/test_lift_pot_reward_wrapper.py`

Reward wrapper follows the composite reward in spec §6. It reads pot pose, target pose, gripper poses, and task-phase flags from `info` (surfaced by the RoboTwin env in Task 13) and outputs a scalar reward.

- [ ] **Step 1: Write the failing test using a mock env**

`tests/unit_tests/test_lift_pot_reward_wrapper.py`:
```python
import numpy as np
import pytest
import gymnasium as gym

from rlinf.envs.robotwin.lift_pot_reward_wrapper import LiftPotSE3RewardWrapper


class _MockLiftPotEnv(gym.Env):
    """Minimal mock: step returns a hand-crafted info dict with SE(3) fields."""

    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, shape=(14,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1, 1, shape=(10,), dtype=np.float32)
        self._step = 0

    def reset(self, **kwargs):
        self._step = 0
        return np.zeros(10, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        info = {
            # Pot pose (identity)
            "pot_pos": np.zeros(3),
            "pot_rot_mat": np.eye(3),
            # Target pose (0.1 m up, 30 deg z rotation)
            "target_pos": np.array([0.0, 0.0, 0.1]),
            "target_rot_mat": np.array(
                [[np.cos(np.pi / 6), -np.sin(np.pi / 6), 0.0],
                 [np.sin(np.pi / 6),  np.cos(np.pi / 6), 0.0],
                 [0.0, 0.0, 1.0]]
            ),
            # Grippers identical to pot for simplicity.
            "left_gripper_pose": np.eye(4),
            "right_gripper_pose": np.eye(4),
            "left_handle_pose": np.eye(4),
            "right_handle_pose": np.eye(4),
            # Task-phase flags.
            "grasp_left_success": True,
            "grasp_right_success": True,
            "lift_distance": 0.06,  # > 5 cm
            "success": False,
        }
        return np.zeros(10, dtype=np.float32), 0.0, False, False, info


def test_se3_reward_components_are_finite():
    env = LiftPotSE3RewardWrapper(_MockLiftPotEnv())
    env.reset()
    _, reward, _, _, info = env.step(env.action_space.sample())
    assert np.isfinite(reward)
    assert "pose_error_pos" in info
    assert "pose_error_rot" in info
    assert info["pose_error_pos"] == pytest.approx(0.1, abs=1e-6)
    assert info["pose_error_rot"] == pytest.approx(np.pi / 6, abs=1e-5)


def test_se3_reward_subtask_progress_both_grasps_plus_lift():
    env = LiftPotSE3RewardWrapper(_MockLiftPotEnv())
    env.reset()
    _, _, _, _, info = env.step(env.action_space.sample())
    # Both grasps + lift > 5 cm → subtask_progress == 1.0.
    assert info["subtask_progress"] == pytest.approx(1.0)


def test_se3_reward_action_smoothness_zero_on_first_step():
    env = LiftPotSE3RewardWrapper(_MockLiftPotEnv())
    env.reset()
    _, _, _, _, info = env.step(np.ones(14, dtype=np.float32))
    assert info["action_rate_penalty"] == pytest.approx(0.0)


def test_se3_reward_action_smoothness_nonzero_on_action_change():
    env = LiftPotSE3RewardWrapper(_MockLiftPotEnv())
    env.reset()
    env.step(np.ones(14, dtype=np.float32))
    _, _, _, _, info = env.step(np.zeros(14, dtype=np.float32))
    assert info["action_rate_penalty"] > 0.0
```

- [ ] **Step 2: Run — should fail with ImportError**

Run:
```bash
pytest tests/unit_tests/test_lift_pot_reward_wrapper.py -v
```

Expected: `ModuleNotFoundError: ... lift_pot_reward_wrapper`.

- [ ] **Step 3: Implement the wrapper**

`rlinf/envs/robotwin/lift_pot_reward_wrapper.py`:
```python
"""SE(3)/SO(3) reward wrapper for RoboTwin lift_pot.

See docs/rob831-project/specs/2026-04-21-robotwin-liftpot-vla-rl-design.md §6
for the reward design.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from rlinf.envs.robotwin.se3_math import se3_log_map, so3_geodesic_distance

__all__ = ["LiftPotSE3RewardWrapper"]


class LiftPotSE3RewardWrapper(gym.Wrapper):
    """Replaces the base env's reward with an SE(3)/SO(3)-based composite.

    The underlying env is expected to surface the following keys in `info`
    on each step (populated by the state-obs plumbing in Task 13):

        pot_pos, pot_rot_mat                    # pot pose
        target_pos, target_rot_mat              # target pose
        left_gripper_pose, right_gripper_pose   # 4x4 SE(3) homogeneous
        left_handle_pose, right_handle_pose     # 4x4 SE(3) homogeneous
        grasp_left_success, grasp_right_success # bool
        lift_distance                           # float, meters
        success                                 # bool
    """

    def __init__(
        self,
        env: gym.Env,
        w_p: float = 1.0,
        w_R: float = 0.3,
        w_ga: float = 0.5,
        w_lift: float = 5.0,
        w_grasp: float = 2.0,
        w_success: float = 10.0,
        w_smooth: float = 0.1,
    ):
        super().__init__(env)
        self.w_p = w_p
        self.w_R = w_R
        self.w_ga = w_ga
        self.w_lift = w_lift
        self.w_grasp = w_grasp
        self.w_success = w_success
        self.w_smooth = w_smooth
        self._prev_action: Optional[np.ndarray] = None

    def reset(self, **kwargs):
        self._prev_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # --- pose errors ---
        pos_err = float(
            np.linalg.norm(np.asarray(info["pot_pos"]) - np.asarray(info["target_pos"]))
        )
        rot_err = so3_geodesic_distance(info["pot_rot_mat"], info["target_rot_mat"])

        # --- gripper-to-handle SE(3) alignment ---
        xi_l = se3_log_map(
            np.linalg.inv(info["left_handle_pose"]) @ info["left_gripper_pose"]
        )
        xi_r = se3_log_map(
            np.linalg.inv(info["right_handle_pose"]) @ info["right_gripper_pose"]
        )
        align_err = float(np.linalg.norm(xi_l) + np.linalg.norm(xi_r))

        # --- task phase ---
        gl = bool(info.get("grasp_left_success", False))
        gr = bool(info.get("grasp_right_success", False))
        lift = float(info.get("lift_distance", 0.0))
        subtask_progress = 0.5 * (float(gl) + float(gr))
        if gl and gr and lift > 0.05:
            subtask_progress = 1.0
        success = bool(info.get("success", False))

        # --- action smoothness ---
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if self._prev_action is None:
            action_rate = 0.0
        else:
            action_rate = float(np.sum((a - self._prev_action) ** 2))
        self._prev_action = a.copy()

        reward = (
            -self.w_p * pos_err
            - self.w_R * rot_err
            - self.w_ga * align_err
            + self.w_lift * max(0.0, lift)
            + self.w_grasp * subtask_progress
            + self.w_success * float(success)
            - self.w_smooth * action_rate
        )

        info.update(
            {
                "pose_error_pos": pos_err,
                "pose_error_rot": rot_err,
                "gripper_handle_alignment": align_err,
                "subtask_progress": subtask_progress,
                "action_rate_penalty": action_rate,
                "reward_se3": reward,
            }
        )
        return obs, reward, terminated, truncated, info
```

- [ ] **Step 4: Run the tests — should pass**

Run:
```bash
pytest tests/unit_tests/test_lift_pot_reward_wrapper.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add rlinf/envs/robotwin/lift_pot_reward_wrapper.py \
        tests/unit_tests/test_lift_pot_reward_wrapper.py
git commit -m "feat(robotwin): add LiftPotSE3RewardWrapper with unit tests"
```

---

## Task 13: Wire the reward wrapper into `RoboTwinEnv`

**Files:**
- Modify: `rlinf/envs/robotwin/robotwin_env.py`
- Create: `examples/embodiment/config/env/robotwin_lift_pot_se3.yaml`

**Pre-task investigation:** the wrapper expects fields like `pot_pos`, `target_rot_mat`, `left_handle_pose`, etc. in the env's `info` dict. RoboTwin's underlying task may or may not expose these. Step 1 is an investigation spike; the wiring that follows depends on what we find.

- [ ] **Step 1: Investigate which pose fields RoboTwin's lift_pot exposes**

Run an interactive probe (inside `rlinf-openvlaoft` conda env):
```bash
python - <<'PY'
import numpy as np
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
from omegaconf import OmegaConf

cfg = OmegaConf.load("examples/embodiment/config/env/robotwin_lift_pot.yaml")
cfg.total_num_envs = 1
cfg.assets_path = "/YOUR/ROBOTWIN/ASSETS"         # set to actual
cfg.seeds_path = "rlinf/envs/robotwin/seeds/train_seeds.json"
cfg.video_cfg.save_video = False
cfg.enable_offload = False

env = RoboTwinEnv(cfg=cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None)
obs, info = env.reset()
print("reset info keys:", list(info.keys()) if isinstance(info, dict) else type(info))
action = env.single_action_space.sample() if hasattr(env, "single_action_space") else np.zeros(14, dtype=np.float32)
obs, rew, term, trunc, info = env.step(action[None, :])
print("step info keys:", list(info.keys()) if isinstance(info, dict) else type(info))
for k, v in (info.items() if isinstance(info, dict) else []):
    if hasattr(v, "shape"):
        print(" ", k, v.shape if hasattr(v, "shape") else type(v))
    else:
        print(" ", k, type(v).__name__)
PY
```

Expected: prints the list of info keys. Record which of the expected fields (`pot_pos`, `pot_rot_mat`, `target_pos`, `target_rot_mat`, `left_gripper_pose`, `right_gripper_pose`, `left_handle_pose`, `right_handle_pose`, `grasp_left_success`, `grasp_right_success`, `lift_distance`, `success`) are already there, which aren't.

Save the findings to `docs/rob831-project/notes/robotwin_info_dict.md`.

- [ ] **Step 2: Depending on findings, augment the RoboTwin env to surface missing fields**

Edit `rlinf/envs/robotwin/robotwin_env.py` — in the `step` method (you'll need to locate the spot where `infos` is built; use `grep -n 'infos\[' rlinf/envs/robotwin/robotwin_env.py` to find it), add whatever is missing by reading from the underlying RoboTwin sim.

Common patterns (concrete code depends on Step 1 findings):
```python
# After the underlying env.step(), before we build our wrapped info dict:
try:
    # RoboTwin typically exposes objects by name via sim.get_actor(name).get_pose()
    pot = self.env.scene.get_actor_by_name("pot")      # or similar
    info["pot_pos"] = np.asarray(pot.pose.p)
    info["pot_rot_mat"] = np.asarray(pot.pose.to_transformation_matrix())[:3, :3]
except Exception:
    # Fall back to a sentinel that the wrapper can detect.
    info.setdefault("pot_pos", np.zeros(3))
    info.setdefault("pot_rot_mat", np.eye(3))
# Repeat for target, grippers, handles, grasp flags, lift distance.
```

If the RoboTwin API doesn't cleanly expose a pose, fall back to reading sim state via `self.env.sim.get_body_pos/quat` or whatever primitive the task uses — document the access path in `notes/robotwin_info_dict.md`.

- [ ] **Step 3: Hook the reward wrapper based on `reward_variant` config**

Add (near the top of `RoboTwinEnv.__init__` or wherever the env is wrapped internally):
```python
self.reward_variant = str(cfg.get("reward_variant", "default")).lower()
# ...later, after the underlying env is built:
if self.reward_variant == "se3":
    from rlinf.envs.robotwin.lift_pot_reward_wrapper import LiftPotSE3RewardWrapper
    # Extract reward weights if present in cfg
    weights = {}
    for k in ["w_p", "w_R", "w_ga", "w_lift", "w_grasp", "w_success", "w_smooth"]:
        if k in cfg:
            weights[k] = float(cfg[k])
    self.env = LiftPotSE3RewardWrapper(self.env, **weights)
```

Exact insertion point depends on the file's current structure — read the `__init__` and locate where the underlying task env is assigned to `self.env`.

- [ ] **Step 4: Create the SE(3) env config variant**

`examples/embodiment/config/env/robotwin_lift_pot_se3.yaml`:
```yaml
# Env variant: RoboTwin lift_pot + SE(3)/SO(3) reward wrapper.

defaults:
  - robotwin_lift_pot

reward_variant: "se3"

# Reward weights (spec §6)
w_p: 1.0
w_R: 0.3
w_ga: 0.5
w_lift: 5.0
w_grasp: 2.0
w_success: 10.0
w_smooth: 0.1
```

- [ ] **Step 5: Integration test — step the env with `reward_variant: se3` and verify `reward_se3` appears in info**

Add to `tests/unit_tests/test_lift_pot_reward_wrapper.py`:
```python
# Skipped when RoboTwin assets / sim aren't installed — marks test as optional.

@pytest.mark.skipif(
    "ROBOTWIN_ASSETS_PATH" not in __import__("os").environ,
    reason="RoboTwin assets not installed",
)
def test_se3_variant_integrates_with_robotwin_env():
    import os
    from omegaconf import OmegaConf
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

    cfg = OmegaConf.load("examples/embodiment/config/env/robotwin_lift_pot_se3.yaml")
    cfg.total_num_envs = 1
    cfg.assets_path = os.environ["ROBOTWIN_ASSETS_PATH"]
    cfg.seeds_path = "rlinf/envs/robotwin/seeds/train_seeds.json"
    cfg.video_cfg.save_video = False
    cfg.enable_offload = False

    env = RoboTwinEnv(cfg=cfg, num_envs=1, seed_offset=0,
                      total_num_processes=1, worker_info=None)
    obs, _ = env.reset()
    action = np.zeros(14, dtype=np.float32)[None, :]
    _, reward, _, _, info = env.step(action)
    assert "reward_se3" in info
    assert np.isfinite(info["reward_se3"])
```

Run (inside conda env, with `ROBOTWIN_ASSETS_PATH` set):
```bash
pytest tests/unit_tests/test_lift_pot_reward_wrapper.py -v
```

Expected: all previous tests pass + the integration test either passes or is skipped if assets aren't on the current machine.

- [ ] **Step 6: Commit**

```bash
git add rlinf/envs/robotwin/robotwin_env.py \
        examples/embodiment/config/env/robotwin_lift_pot_se3.yaml \
        tests/unit_tests/test_lift_pot_reward_wrapper.py \
        docs/rob831-project/notes/robotwin_info_dict.md
git commit -m "feat(robotwin): wire SE(3) reward wrapper via reward_variant config"
```

---

## Task 14: Verify M1 convergence (decision gate)

**Files:** none (analysis).

By now M1 has been running since Task 9. Check it.

- [ ] **Step 1: Check M1 status**

Run:
```bash
squeue -u $USER -j <M1_JOB_ID>
tail -n 50 slurm/logs/m1_vla_grpo_<M1_JOB_ID>.out
```

Expected: Either still running with eval checkpoints every 10 epochs, or completed.

- [ ] **Step 2: Pull the eval-success curve from TensorBoard**

Run:
```bash
python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, os
path = glob.glob(os.path.expanduser('~/OCEANDIR/projects/RLinf/../results/m1_vla_grpo_*/tensorboard'))[-1]
acc = EventAccumulator(path); acc.Reload()
for tag in acc.Tags()['scalars']:
    if 'success' in tag or 'return' in tag:
        evs = acc.Scalars(tag)
        if evs:
            print(tag, [(e.step, round(e.value, 3)) for e in evs[-10:]])
"
```

Expected: `eval/success_once` values improving over time (or at least not regressing).

- [ ] **Step 3: Apply the spec's M1 decision gate**

- If after ~30% of max_epochs (epoch ≥ 45), eval success is *below* B4's zero-shot number: the single-GPU scaling is not working. Options:
  - Increase `global_batch_size` (raise `grad_accum_steps`).
  - Warm-start from a higher LoRA rank (16 → 32).
  - Reduce learning rate (e.g., 2e-4 → 5e-5).
  - If none work, flag the issue in the report and treat M1 as a negative result.
- If eval success trending upward: let M1 run to completion. Proceed to Task 15 to prepare M2b.

- [ ] **Step 4: Commit the findings note**

```bash
cat > docs/rob831-project/results/m1_vla_grpo.md <<EOF
# M1 — VLA-GRPO (1-GPU) Results

- Job: \`m1_vla_grpo_<JOB_ID>\`
- B4 baseline success: X.XX
- M1 eval success trajectory (last 10 evals): ...
- Final success: X.XX
- Gate decision: continuing / pivoting / flagged as negative

EOF
git add docs/rob831-project/results/m1_vla_grpo.md
git commit -m "docs(rob831): log M1 VLA-GRPO 1-GPU training results"
```

---

## Task 15: Stage 3 — M2b config + SLURM script

**Files:**
- Create: `examples/embodiment/config/robotwin_lift_pot_grpo_openvlaoft_1gpu_se3.yaml`
- Create: `slurm/robo/m2b_vla_grpo_se3.sh`

- [ ] **Step 1: Write the M2b config**

```yaml
# M2b: VLA-GRPO 1-GPU with SE(3)/SO(3) reward wrapper.
# Same as M1 except env switches to the SE(3) variant.

defaults:
  - robotwin_lift_pot_grpo_openvlaoft_1gpu
  - env/robotwin_lift_pot_se3@env.train
  - env/robotwin_lift_pot_se3@env.eval
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null

runner:
  logger:
    experiment_name: "m2b_vla_grpo_se3_1gpu"
```

- [ ] **Step 2: Verify parse**

Run:
```bash
EMBODIED_PATH=examples/embodiment/ REPO_PATH=$PWD \
ROBOTWIN_ASSETS_PATH=/tmp/_fake VLA_CKPT_PATH=/tmp/_fake \
python -c "
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
with initialize_config_dir(version_base='1.1', config_dir='$PWD/examples/embodiment/config'):
    cfg = compose(config_name='robotwin_lift_pot_grpo_openvlaoft_1gpu_se3')
print('reward_variant:', cfg.env.train.reward_variant)
print('w_R:', cfg.env.train.w_R)
"
```

Expected: `reward_variant: se3`, `w_R: 0.3`.

- [ ] **Step 3: Write the SLURM script (copy m1 and swap config name)**

```bash
# slurm/robo/m2b_vla_grpo_se3.sh
#!/bin/bash
#SBATCH --job-name=m2b_vla_grpo_se3
#SBATCH --partition=ROBO
#SBATCH --reservation=ROBOcis220039p
#SBATCH --nodelist=robo-gh005
#SBATCH --gres=gpu:h100:1
#SBATCH --time=16:00:00
#SBATCH -A cis220039p
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

cd "${REPO_PATH}"
python examples/embodiment/train_embodied_agent.py \
  --config-path=examples/embodiment/config/ \
  --config-name=robotwin_lift_pot_grpo_openvlaoft_1gpu_se3 \
  runner.logger.log_path="${RESULTS_DIR}"
```

- [ ] **Step 4: Commit**

```bash
chmod +x slurm/robo/m2b_vla_grpo_se3.sh
git add examples/embodiment/config/robotwin_lift_pot_grpo_openvlaoft_1gpu_se3.yaml \
        slurm/robo/m2b_vla_grpo_se3.sh
git commit -m "feat(config): add M2b VLA-GRPO + SE(3) reward 1-GPU config and SLURM script"
```

---

## Task 16: Submit M2b and verify

**Files:** none (results go to `docs/rob831-project/results/`).

- [ ] **Step 1: Submit M2b**

Run (after M1 has finished — the reservation only allows one job at a time):
```bash
sbatch slurm/robo/m2b_vla_grpo_se3.sh
```

Expected: `Submitted batch job <JOB_ID>`. Record the ID.

Alternatively, if an additional H100 is available outside the reservation (non-`ROBO` partition), you can submit M2b in parallel — but don't fight for the reservation node.

- [ ] **Step 2: Monitor to completion and extract eval curve**

Same pattern as Task 6:
```bash
tail -f slurm/logs/m2b_vla_grpo_se3_<JOB_ID>.out
```

When done:
```bash
python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, os
path = glob.glob(os.path.expanduser('~/OCEANDIR/projects/RLinf/../results/m2b_vla_grpo_se3_*/tensorboard'))[-1]
acc = EventAccumulator(path); acc.Reload()
for tag in acc.Tags()['scalars']:
    if 'success' in tag:
        evs = acc.Scalars(tag)
        if evs:
            print(tag, round(evs[-1].value, 3))
"
```

- [ ] **Step 3: Log results**

```bash
cat > docs/rob831-project/results/m2b_vla_grpo_se3.md <<EOF
# M2b — VLA-GRPO + SE(3) Reward Results

- Job: \`m2b_vla_grpo_se3_<JOB_ID>\`
- Compared to M1 (default RoboTwin reward): Δsuccess = ...
- Final eval success: X.XX
- Eval videos: \`results/m2b_vla_grpo_se3_<JOB_ID>/video/eval/\`
EOF
git add docs/rob831-project/results/m2b_vla_grpo_se3.md
git commit -m "docs(rob831): log M2b VLA-GRPO + SE(3) reward results"
```

---

## Task 17: VLA-track summary plot

**Files:**
- Modify: `scripts/plot_return.py` if needed, or add a new standalone plot script.

- [ ] **Step 1: Verify `scripts/plot_return.py` already handles multiple experiments**

Read:
```bash
head -60 scripts/plot_return.py
```

Expected: the script accepts repeated `--results` / `--name` flags (it was rewritten in PR 3 for exactly this purpose).

- [ ] **Step 2: Generate the combined plot**

Run:
```bash
python scripts/plot_return.py \
  --results ../results/b4_vla_zeroshot_<B4_JOB_ID>/tensorboard \
  --name "B4 zero-shot" \
  --results ../results/m1_vla_grpo_<M1_JOB_ID>/tensorboard \
  --name "M1 VLA-GRPO" \
  --results ../results/m2b_vla_grpo_se3_<M2B_JOB_ID>/tensorboard \
  --name "M2b VLA-GRPO + SE(3)" \
  --output docs/rob831-project/results/vla_track_success.png \
  --title "VLA track: success rate vs env steps"
```

Expected: a PNG is written at the output path. Sanity-check that the B4 line is flat at its zero-shot value (no training), and M1/M2b curves show RL progression.

- [ ] **Step 3: Commit**

```bash
git add docs/rob831-project/results/vla_track_success.png
git commit -m "docs(rob831): add VLA-track summary plot (B4 vs M1 vs M2b)"
```

---

## End of Plan 1

At this point we have:

- B4 (zero-shot VLA) success number.
- M1 (VLA + GRPO) success curve + final number.
- M2b (VLA + GRPO + SE(3) reward) success curve + final number.
- SE(3) reward wrapper + Lie-group utilities, unit-tested.
- 4 committed result summaries and 1 plot.

**Plan 2** (to be written after M1 results arrive) covers:
- Stage 4: state-obs mode in the RoboTwin env.
- Stage 5: B1 (MLP-PPO) and B2 (MLP-SAC) configs + runs.
- Stage 6: M2a (MLP-SAC + SE(3)) run.
- Stage 7: MBPO audit and B3 (conditional).
- Stage 8: Final writeup.

The decomposition is intentional — Plan 2's configurations will be informed by what we learn about the RoboTwin env during Task 13's investigation, and by the M1 convergence behavior (which may shift our expectations for MLP baselines).
