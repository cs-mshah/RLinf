#!/bin/bash
#SBATCH --job-name=b1_roboeval_ppo
#SBATCH --partition=ROBO
#SBATCH --reservation=ROBOcis220039p
#SBATCH --nodelist=robo-gh005
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:2
#SBATCH --time=08:00:00
#SBATCH -A cis220039p
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

source "${SLURM_SUBMIT_DIR}/slurm/robo/_common.sh"

python examples/embodiment/train_embodied_agent.py \
  --config-path="${REPO_PATH}/examples/embodiment/config/" \
  --config-name=roboeval_liftpot_ppo_mlp_v2 \
  cluster.num_nodes=1 \
  runner.logger.log_path="${RESULTS_DIR}"

echo "[done] B1 (RoboEval PPO v2) completed. Results in ${RESULTS_DIR}."
