#!/bin/bash
set -o xtrace


################################################################################
# Utilities
################################################################################
# Different terminal styling constants
GREEN="\e[32m"
RED="\033[31m"
BLUE="\e[34m"
BOLD="\e[1m"
RESET_ALL="\e[0m"
RESET_FG="\e[39m"


################################################################################
# Define constants
################################################################################
export FAISS_INDEX_FACTORY="PCAR128,IVF16384,SQfp16"
GPU_TYPE=turing
PARTITION=long
export PROJECT_ROOT="$HOME/ELI5/"
export CREATE_NP_MEMMAP=true
export CREATE_DPR_EMBEDDINGS=true
export DPR_EMBEDDING_DEPTH=128


# echo -e "GPU_TYPE:            $BLUE$BOLD$GPU_TYPE$RESET_ALL"
# echo -e "PARTITION:           $BLUE$BOLD$PARTITION$RESET_ALL"
# echo -e "FAISS_INDEX_FACTORY: $BLUE$BOLD$FAISS_INDEX_FACTORY$RESET_ALL"
# echo -e "PROJECT_ROOT:        $BLUE$BOLD$PROJECT_ROOT$RESET_ALL"


################################################################################
# Run job(s)
################################################################################
TIMESTAMP="$(date +'%Y-%m-%d_%H-%M-%S')"
sbatch \
--output=/home/mila/g/gagnonju/logs/out_"$FAISS_INDEX_FACTORY"_"$TIMESTAMP".txt \
 --error=/home/mila/g/gagnonju/logs/err_"$FAISS_INDEX_FACTORY"_"$TIMESTAMP".txt \
--partition="$PARTITION" \
--export=ALL \
--mem=128Gb \
--gres=gpu:"$GPU_TYPE":1 \
-J="$FAISS_INDEX_FACTORY" \
--ntasks=1 \
--time=48:00:00 \
"$PROJECT_ROOT"/run_baseline_pretrained.sh

# --export=FAISS_INDEX_FACTORY="$FAISS_INDEX_FACTORY" \
# --export=PROJECT_ROOT="$PROJECT_ROOT" \

set +o xtrace