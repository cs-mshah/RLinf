#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=sac_mlp_policy
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:2
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=4
#SBATCH -A cis260009p

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

# SAC MLP policy for RoboEval LiftPot
RAY_ADDRESS=local MUJOCO_GL=egl EMBODIED_PATH=examples/embodiment/ \
  python examples/embodiment/train_embodied_agent.py \
  --config-name roboeval_liftpot_sac_mlp \
  cluster.num_nodes=1 \
  runner.logger.log_path="${RESULTS_DIR}"

# Construct log file names using SLURM environment variables
OUTPUT_FILE="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
ERROR_FILE="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

# Move the log files to LOG_DIR after job execution
echo "Moving log files to ${LOG_DIR}"
mv ${OUTPUT_FILE} ${LOG_DIR}/
mv ${ERROR_FILE} ${LOG_DIR}/
