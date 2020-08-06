#!/bin/bash


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
    --dpr_faiss_path="$PROJECT_ROOT/saves/$FAISS_INDEX_FACTORY.faiss" \
    --create_faiss_dpr=True

