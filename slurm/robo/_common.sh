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
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

# External paths — override in your submit env or .bashrc if different.
export ROBOTWIN_ASSETS_PATH=${ROBOTWIN_ASSETS_PATH:-$HOME/OCEANDIR/data/robotwin_assets}
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
