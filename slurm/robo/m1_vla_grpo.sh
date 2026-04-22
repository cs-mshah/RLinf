#!/bin/bash
#SBATCH --job-name=m1_vla_grpo
#SBATCH --partition=ROBO
#SBATCH --reservation=ROBOcis220039p
#SBATCH --nodelist=robo-gh005
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --gres=gpu:h100:2
#SBATCH --time=16:00:00
#SBATCH -A cis220039p
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

source "${SLURM_SUBMIT_DIR}/slurm/robo/_common.sh"

python examples/embodiment/train_embodied_agent.py \
  --config-path="${REPO_PATH}/examples/embodiment/config/" \
  --config-name=robotwin_lift_pot_grpo_openvlaoft_1gpu \
  runner.logger.log_path="${RESULTS_DIR}"

echo "[done] M1 training completed. Results in ${RESULTS_DIR}."
