#!/bin/bash
#SBATCH --job-name=download_vla_ckpt
#SBATCH --partition=ROBO
#SBATCH --reservation=ROBOcis220039p
#SBATCH --nodelist=robo-gh005
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH -A cis220039p
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Downloads the OpenVLA-OFT SFT checkpoint for RoboTwin lift_pot from HuggingFace.
# Destination: $HOME/OCEANDIR/checkpoints/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot

set -eo pipefail
module load anaconda3
source /ocean/projects/cis220039p/hkwon/projects/RLinf/.venv/bin/activate

CKPT_DIR=${VLA_CKPT_PATH:-$HOME/OCEANDIR/checkpoints/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot}
mkdir -p "$(dirname "${CKPT_DIR}")"

echo "[ckpt] downloading to ${CKPT_DIR}"
huggingface-cli download RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot \
  --local-dir "${CKPT_DIR}"

echo "[ckpt] done. Contents:"
ls -la "${CKPT_DIR}" | head -20
echo "[ckpt] size: $(du -sh "${CKPT_DIR}" | cut -f1)"
