#!/bin/bash
#SBATCH --job-name=build_rlinf_openvlaoft
#SBATCH --partition=ROBO
#SBATCH --reservation=ROBOcis220039p
#SBATCH --nodelist=robo-gh005
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH -A cis220039p
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# One-shot: build the .venv/ python environment for RLinf + openvla-oft + robotwin
# on the reserved GH-node (CPU build; no GPU needed for install).
#
# Output venv goes to $REPO_PATH/.venv (install.sh default).

set -eo pipefail

REPO_PATH=/ocean/projects/cis220039p/hkwon/projects/RLinf
cd "${REPO_PATH}"

echo "[build_env] host=$(hostname)"
echo "[build_env] start=$(date -Iseconds)"

# --- Modules ------------------------------------------------------------
# NOTE: default PSC gcc (8.x) is too old for PyTorch C++ extension builds
# (pytorch3d + curobo rebuild CUDA kernels and require gcc >= 9). Load gcc/10.2.0.
module load anaconda3 cuda gcc/10.2.0
export CC=$(command -v gcc)
export CXX=$(command -v g++)
echo "[build_env] loaded anaconda3, cuda, gcc/10.2.0"
echo "[build_env] CC=${CC} (gcc $(gcc -dumpversion))"
echo "[build_env] CXX=${CXX} (g++ $(g++ -dumpversion))"
echo "[build_env] nvcc: $(command -v nvcc) ($(nvcc --version 2>&1 | tail -1))"

# --- Bootstrap uv (the install.sh dependency) ---------------------------
# install.sh requires `uv` on PATH; if it's not already installed, add it
# to the user bin via pip. Anaconda3 provides pip.
if ! command -v uv >/dev/null 2>&1; then
    echo "[build_env] uv not found, installing via pip..."
    python3 -m pip install --user --upgrade uv
    export PATH="$HOME/.local/bin:$PATH"
    hash -r
fi
echo "[build_env] uv: $(command -v uv) ($(uv --version 2>/dev/null || echo 'version-unknown'))"

# --- Clean any partial .venv from a previous failed run -----------------
if [ -d "${REPO_PATH}/.venv" ]; then
    echo "[build_env] removing existing .venv for a clean rebuild"
    rm -rf "${REPO_PATH}/.venv"
fi

# --- Run the repo's install script --------------------------------------
# --no-root: we're not root on PSC (skips system-package apt steps).
# install.sh creates ./.venv and installs openvla-oft + robotwin extras there.
echo "[build_env] running: bash requirements/install.sh embodied --model openvla-oft --env robotwin --no-root"
bash requirements/install.sh embodied --model openvla-oft --env robotwin --no-root

# --- Sanity checks ------------------------------------------------------
echo "[build_env] activating .venv for sanity checks"
source .venv/bin/activate

python -c "import sys; print('python', sys.version)"
python -c "import torch; print('torch', torch.__version__, 'cuda-avail:', torch.cuda.is_available())"
python -c "import rlinf; print('rlinf OK:', rlinf.__file__)"
python -c "from rlinf.models.embodiment.openvla_oft.official import __init__ as _; print('openvla_oft OK')" 2>&1 | tail -5 || echo "[warn] openvla_oft import probe failed — inspect log"
python -c "import robotwin; print('robotwin OK:', robotwin.__file__)" 2>&1 | tail -5 || echo "[warn] robotwin import probe failed — inspect log"

echo "[build_env] done=$(date -Iseconds)"
echo "[build_env] venv at: ${REPO_PATH}/.venv"
echo "[build_env] activate with: source ${REPO_PATH}/.venv/bin/activate"
