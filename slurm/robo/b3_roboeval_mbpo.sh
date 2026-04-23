#!/bin/bash
#SBATCH --job-name=b3_roboeval_mbpo
#SBATCH --partition=ROBO
#SBATCH --reservation=ROBOcis220039p
#SBATCH --nodelist=robo-gh005
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH -A cis220039p
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

source "${SLURM_SUBMIT_DIR}/slurm/robo/_common.sh"

mkdir -p "${RESULTS_DIR}/tensorboard"

python scripts/b3_mbpo_roboeval.py \
  --config examples/embodiment/config/env/roboeval_liftpot_state.yaml \
  --out "${RESULTS_DIR}/tensorboard" \
  --total-steps 200000 \
  --eval-every 5000 \
  --train-envs 16 \
  --eval-envs 8

echo "[done] B3 (MBPO) completed. Results in ${RESULTS_DIR}."
