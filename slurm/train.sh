#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=mlp_policy
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:2
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=4
#SBATCH -A cis260009p

# You can request at most 4 GPUs from one node in the GPU-shared partition.
# type: v100-16, v100-32, l40s-48, and h100-80
# -n n Number of cores requested in total.
# --ntasks-per-node=n # Request n cores be allocated per node.

# set -x

module load anaconda3
module load cuda

source /jet/home/mshah10/miniconda3/etc/profile.d/conda.sh
conda activate roboeval

# Results go into results/<experiment_name>/
EXPERIMENT_NAME="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
RESULTS_DIR="../results/${EXPERIMENT_NAME}"

echo "JOB_NAME: ${SLURM_JOB_NAME}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "RESULTS_DIR: ${RESULTS_DIR}"

# MLP policy (recommended)
MUJOCO_GL=egl EMBODIED_PATH=/path/to/examples/embodiment \
  python examples/embodiment/train_embodied_agent.py \
  --config-name roboeval_liftpot_ppo_mlp \
  cluster.num_nodes=1 \
  runner.logger.log_path="${RESULTS_DIR}"

# # CNN policy
# MUJOCO_GL=egl EMBODIED_PATH=/path/to/examples/embodiment \
#   python examples/embodiment/train_embodied_agent.py \
#   --config-name roboeval_liftpot_ppo_cnn \
#   cluster.num_nodes=1 \
#   'cluster.component_placement={actor,env,rollout: "0,1"}' \
#   actor.model.model_path=/path/to/RLinf-ResNet10-pretrained


# Construct log file names using SLURM environment variables
OUTPUT_FILE="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
ERROR_FILE="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

# Move the log files to LOG_DIR after job execution
echo "Moving log files to ${LOG_DIR}"
mv ${OUTPUT_FILE} ${LOG_DIR}/
mv ${ERROR_FILE} ${LOG_DIR}/