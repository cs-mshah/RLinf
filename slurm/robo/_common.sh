# slurm/robo/_common.sh
# Sourced by each robo/*.sh SLURM script. Defines common paths and env vars.

set -eo pipefail

module load anaconda3 cuda

# Resolve REPO_PATH from either BASH_SOURCE (when sourced from a checked-out
# script) or SLURM_SUBMIT_DIR (when sbatch doesn't preserve BASH_SOURCE).
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
    export REPO_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    export REPO_PATH="${SLURM_SUBMIT_DIR}"
else
    echo "ERROR: cannot resolve REPO_PATH — export it manually before sourcing _common.sh" >&2
    exit 1
fi

cd "${REPO_PATH}"

# Activate the uv-managed venv built by slurm/robo/build_env.sh.
if [ ! -f "${REPO_PATH}/.venv/bin/activate" ]; then
    echo "ERROR: ${REPO_PATH}/.venv not found. Run slurm/robo/build_env.sh first." >&2
    exit 1
fi
source "${REPO_PATH}/.venv/bin/activate"

export EMBODIED_PATH="${REPO_PATH}/examples/embodiment/"
export MUJOCO_GL=egl
# sapien uses Vulkan for rendering. Vulkan 1.3+ reads VK_DRIVER_FILES first and
# falls back to VK_ICD_FILENAMES. PSC pre-sets *both* to /etc/vulkan/icd.d/nvidia_icd.json
# which doesn't exist on robo-gh005 — only /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
# does. Unset both and pick the first existing candidate for each.
unset VK_ICD_FILENAMES VK_DRIVER_FILES
for _vk_icd in /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
    if [ -f "$_vk_icd" ]; then
        export VK_ICD_FILENAMES="$_vk_icd"
        export VK_DRIVER_FILES="$_vk_icd"
        break
    fi
done
# PyOpenGL fallback (matches examples/embodiment/run_embodiment.sh).
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
# ROBOT_PLATFORM selects action-dim/normalization preset for the VLA.
# For RoboTwin PIPER bimanual (14-DOF), ALOHA is the closest-fitting preset.
export ROBOT_PLATFORM=${ROBOT_PLATFORM:-ALOHA}
export HYDRA_FULL_ERROR=1
# Force a per-job isolated Ray cluster. The ROBO node is shared; if two Ray
# instances co-exist under /tmp/ray/sessions, Ray's state API aborts with
# "Found multiple active Ray instances". Use a unique tmp dir per SLURM
# job so each job has its own Ray session.
export RAY_ADDRESS=local
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID:-$$}"
mkdir -p "${RAY_TMPDIR}"

# RoboTwin repo — must be on PYTHONPATH; the RLinf install script does NOT
# clone it (it only installs RoboTwin's pip-level deps: sapien, mplib, curobo).
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-$HOME/OCEANDIR/projects/RoboTwin}
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}"

# External paths — override in your submit env or .bashrc if different.
# IMPORTANT: ROBOTWIN_ASSETS_PATH is the RoboTwin REPO ROOT, not the `assets/`
# subdirectory. RoboTwin's own code (envs/utils/rand_create_cluttered_actor.py:22)
# computes `BASE_DIR / "assets/..."` — if we pass `.../RoboTwin/assets` here it
# doubles to `.../RoboTwin/assets/assets/...` and the loader crashes.
export ROBOTWIN_ASSETS_PATH=${ROBOTWIN_ASSETS_PATH:-$ROBOTWIN_PATH}
export VLA_CKPT_PATH=${VLA_CKPT_PATH:-$HOME/OCEANDIR/checkpoints/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot}

export EXPERIMENT_NAME="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
export RESULTS_DIR="${REPO_PATH}/../results/${EXPERIMENT_NAME}"
mkdir -p "${RESULTS_DIR}"

echo "[_common] host=$(hostname)"
echo "[_common] REPO_PATH=${REPO_PATH}"
echo "[_common] VLA_CKPT_PATH=${VLA_CKPT_PATH}"
echo "[_common] ROBOTWIN_ASSETS_PATH=${ROBOTWIN_ASSETS_PATH}"
echo "[_common] RESULTS_DIR=${RESULTS_DIR}"
echo "[_common] python=$(which python)"
