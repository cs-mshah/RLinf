#!/bin/bash
#SBATCH --job-name=install_roboeval
#SBATCH --partition=ROBO
#SBATCH --reservation=ROBOcis220039p
#SBATCH --nodelist=robo-gh005
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH -A cis220039p
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Install the roboeval Python package (cloned at $HOME/OCEANDIR/projects/RoboEval)
# into our existing .venv. Use --no-deps to avoid clobbering our openvla-oft /
# robotwin versions of gymnasium / numpy / mujoco.

set -eo pipefail
module load anaconda3
source /ocean/projects/cis220039p/hkwon/projects/RLinf/.venv/bin/activate

echo "[install] pip: $(which pip)"
cd /jet/home/hkwon/OCEANDIR/projects/RoboEval

echo "[install] installing roboeval --no-deps ..."
pip install -e . --no-deps

echo "[install] installing mojo dep (needed by roboeval.action_modes) ..."
pip install --no-deps "mojo @ git+https://github.com/helen9975/mojo.git"

echo "[install] installing other light deps that roboeval needs ..."
# these are small, low-conflict packages that roboeval uses
pip install --no-deps dm_control==1.0.31 pyquaternion mujoco_utils wget dearpygui click-prompt rich-click dm_env || true

echo "[install] verifying imports ..."
python -c "
import roboeval
print('roboeval:', roboeval.__file__)
from roboeval.action_modes import JointPositionActionMode
print('action_modes OK')
from roboeval.envs.lift_pot import LiftPot
print('lift_pot OK')
"

echo "[install] done."
