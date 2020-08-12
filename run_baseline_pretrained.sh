#!/bin/bash
################################################################################
# This script is not meant to be run directly. It is meant to be sbatched,
# and forwards the arguments it receives from a seperate launcher script to 
# the python script. 
################################################################################


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
NP_MEMMAP_NAME="dpr_np_memmap"
EXTRA_ARGS=
if [[ ! -z $CREATE_DPR_EMBEDDINGS ]] ; then
    # We are creating our own embeddings
    EXTRA_ARGS="--create_dpr_embeddings=True $EXTRA_ARGS"
    EXTRA_ARGS="--nproc=1 $EXTRA_ARGS"
fi
if [[ ! -z $CREATED_DPR_EMBEDDINGS ]] ; then
    # We are using the embeddings we create or are about to create
    FAISS_NAME="created_embs_$FAISS_NAME"
    NP_MEMMAP_NAME="created_embs_$NP_MEMMAP_NAME"
fi
if [[ ! -z $CREATE_NP_MEMMAP ]] ; then
    # We are creating the memmap
    EXTRA_ARGS="--create_np_memmap=True $EXTRA_ARGS"
else
    EXTRA_ARGS="--create_np_memmap=False $EXTRA_ARGS"
fi
if [[ ! -z $DPR_EMBEDDING_DEPTH ]] ; then
    EXTRA_ARGS="--dpr_embedding_depth=$DPR_EMBEDDING_DEPTH $EXTRA_ARGS"
else
    EXTRA_ARGS="--dpr_embedding_depth=768 $EXTRA_ARGS"
fi

echo -e "EXTRA_ARGS: $BOLD$BLUE$EXTRA_ARGS$RESET_ALL"
echo -e "NP_MEMMAP_NAME: $BOLD$BLUE$NP_MEMMAP_NAME$RESET_ALL"
echo -e "FAISS_NAME: $BOLD$BLUE$FAISS_NAME$RESET_ALL"


################################################################################
# Checks
################################################################################
if [[ ! -z "$CREATE_DPR_EMBEDDINGS" ]] && [[ -z $CREATED_DPR_EMBEDDINGS ]] ;
then 
    echo -e "$RED${BOLD}CREATE_DPR_EMBEDDINGS is true but CREATED_DPR_EMBEDDINGS is false."
    echo -e "This doesn't make sense."
    echo -e "We can't ask to create DPR embeddings without using DPR embeddings."
    exit 1
fi
if [[ -z "$CREATE_DPR_EMBEDDINGS" ]] && [[ ! -z $CREATED_DPR_EMBEDDINGS ]] && [[ ! -z $CREATE_NP_MEMMAP ]];
then 
    echo -e "$RED${BOLD}CREATE_DPR_EMBEDDINGS is false but CREATED_DPR_EMBEDDINGS is True and CREATE_NP_MEMMAP is true."
    echo -e "This doesn't make sense."
    echo -e "We are asking to create the DPR memmap and to use created embeddings "
    echo -e "but not saying we are creating the embeddings."
    exit 1
fi
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
