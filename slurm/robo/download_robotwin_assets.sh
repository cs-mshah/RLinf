#!/bin/bash
#SBATCH --job-name=download_robotwin_assets
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

set -eo pipefail
module load anaconda3
source /ocean/projects/cis220039p/hkwon/projects/RLinf/.venv/bin/activate

ROBOTWIN_DIR=/jet/home/hkwon/OCEANDIR/projects/RoboTwin
echo "[assets] downloading to ${ROBOTWIN_DIR}/assets"
cd "${ROBOTWIN_DIR}/assets"
python _download.py
echo "[assets] unzipping..."
for z in background_texture.zip embodiments.zip objects.zip; do
    if [ -f "$z" ]; then
        echo "  unzipping $z..."
        unzip -q -o "$z" && rm "$z"
    fi
done
echo "[assets] done. Final tree:"
ls -la "${ROBOTWIN_DIR}/assets" | head -20
