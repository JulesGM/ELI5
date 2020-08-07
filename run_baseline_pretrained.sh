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
# Deal with potential extra args
################################################################################
FAISS_NAME="$FAISS_INDEX_FACTORY"
NP_MEMMAP_NAME="dpr_np_memmap.dat"
EXTRA_ARGS=
if $CREATE_DPR_EMBEDDINGS; then
    EXTRA_ARGS="--create_dpr_embeddings=True $EXTRA_ARGS"
    FAISS_NAME="created_$FAISS_NAME"
    NP_MEMMAP_NAME="created_$NP_MEMMAP_NAME"
fi
if $DPR_EMBEDDING_DEPTH; then
    EXTRA_ARGS="--dpr_embedding_depth=$DPR_EMBEDDING_DEPTH $EXTRA_ARGS"
fi
if $CREATE_NP_MEMMAP; then
    EXTRA_ARGS="--create_np_memmap=True $EXTRA_ARGS"
fi
echo -e "EXTRA_ARGS: `$BOLD$BLUE$EXTRA_ARGS$RESET_ALL`"
echo -e "NP_MEMMAP_NAME: `$BOLD$BLUE$NP_MEMMAP_NAME$RESET_ALL`"
echo -e "FAISS_NAME: `$BOLD$BLUE$FAISS_NAME$RESET_ALL`"


################################################################################
# Checks
################################################################################
if [[ -z $FAISS_INDEX_FACTORY ]] ; then
    echo ">>> FAISS_INDEX_FACTORY is unset, quitting."
    exit
else
    echo -e "FAISS_INDEX_FACTORY: $BLUE$BOLD$FAISS_INDEX_FACTORY$RESET_ALL"
fi

if [[ -z $PROJECT_ROOT ]] ; then
    echo ">>> PROJECT_ROOT is unset, quitting."
    exit
else
    echo -e "PROJECT_ROOT: $BLUE$BOLD$PROJECT_ROOT$RESET_ALL"
fi


################################################################################
# Load the environment
################################################################################
echo "Loading env:"
source "$PROJECT_ROOT/load_env.sh"


################################################################################
# Run the script
################################################################################
echo "Running script:"
python "$PROJECT_ROOT/baseline_pretrained.py" \
    --faiss_index_factory="$FAISS_INDEX_FACTORY" \
    --dpr_faiss_path="$PROJECT_ROOT/saves/$FAISS_NAME.faiss" \
    --dpr_np_memmmap_path="$PROJECT_ROOT/saves/$NP_MEMMAP_NAME.dat" \
    --create_faiss_dpr=True \
    $EXTRA_ARGS

set +o xtrace