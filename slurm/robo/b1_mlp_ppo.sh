#!/bin/bash
#SBATCH --job-name=b1_mlp_ppo
#SBATCH --partition=ROBO
#SBATCH --reservation=ROBOcis220039p
#SBATCH --nodelist=robo-gh005
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=06:00:00
#SBATCH -A cis220039p
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

source "${SLURM_SUBMIT_DIR}/slurm/robo/_common.sh"

python examples/embodiment/train_embodied_agent.py \
  --config-path="${REPO_PATH}/examples/embodiment/config/" \
  --config-name=robotwin_lift_pot_ppo_mlp \
  runner.logger.log_path="${RESULTS_DIR}"

echo "[done] B1 training completed. Results in ${RESULTS_DIR}."
