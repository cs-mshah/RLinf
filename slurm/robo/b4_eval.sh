#!/bin/bash
#SBATCH --job-name=b4_vla_zeroshot
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

python examples/embodiment/eval_embodied_agent.py \
  --config-path=examples/embodiment/config/ \
  --config-name=robotwin_lift_pot_grpo_openvlaoft_eval_1gpu \
  runner.logger.log_path="${RESULTS_DIR}"

echo "[done] B4 eval completed. Results in ${RESULTS_DIR}."
