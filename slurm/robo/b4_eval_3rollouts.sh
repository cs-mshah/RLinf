#!/bin/bash
#SBATCH --job-name=b4_vla_zeroshot_3r
#SBATCH --partition=ROBO
#SBATCH --reservation=ROBOcis220039p
#SBATCH --nodelist=robo-gh005
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH -A cis220039p
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

source "${SLURM_SUBMIT_DIR}/slurm/robo/_common.sh"

# 3 rollouts × 16 envs = 48 trajectories. Tightens variance on the B4 number.
python examples/embodiment/eval_embodied_agent.py \
  --config-path="${REPO_PATH}/examples/embodiment/config/" \
  --config-name=robotwin_lift_pot_grpo_openvlaoft_eval_1gpu \
  runner.logger.log_path="${RESULTS_DIR}" \
  algorithm.eval_rollout_epoch=3

echo "[done] B4 (3-rollout) eval completed. Results in ${RESULTS_DIR}."
