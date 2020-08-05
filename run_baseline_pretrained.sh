#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/home/mila/g/gagnonju/logs/out_PCAR32,IVF65536_HNSW32,SQfp16.txt
#SBATCH --error=/home/mila/g/gagnonju/logs/err_PCAR32,IVF65536_HNSW32,SQfp16.txt
#SBATCH --ntasks=1
#SBATCH --time=48:00
#SBATCH --gres=gpu:turing:1
#SBATCH --partition=long


###############################################################################
# Utilities
###############################################################################
# Different terminal styling constants
GREEN="\e[32m"
RED="\033[31m"
BLUE="\e[34m"
BOLD="\e[1m"
RESET_ALL="\e[0m"
RESET_FG="\e[39m"

PROJECT_ROOT="$HOME/ELI5/"


################################################################################
# Load the environment
################################################################################
echo "Project root: $PROJECT_ROOT"
echo "Loading env:"
source "$PROJECT_ROOT/load_env.sh"


################################################################################
# Run the script
################################################################################
echo "Running script:"

python "$PROJECT_ROOT/baseline_pretrained.py" \
    --faiss_index_factory="PCAR32,IVF65536_HNSW32,SQfp16"

