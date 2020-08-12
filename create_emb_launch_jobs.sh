#!/bin/bash
# From https://github.com/mattbryson/bash-arg-parse/blob/master/arg_parse_example
# And https://stackoverflow.com/a/14203146/447599


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
# Parse Arguments
################################################################################
function parse_args
{
    POSITIONAL=()
    while [[ $# -gt 0 ]]
    do
        case "$1" in
            -i|--interactive)
            # Wether we are running interactive mode instead of batch mode.
            INTERACTIVE=true
            shift 
            ;;
            
            -upme|--use_premaid_embs)
            # Whether to use embedding we generated ourselves or to use 
            # huggingface's premaid ones.
            USE_PREMAID_EMBS=true
            shift
            ;;

            *)
            echo -e "$RED${BOLD}Unsupported arg: \`$1\`$RESET_ALL"
            exit 1
            # POSITIONAL+=("$1")
            shift 
            ;;
        esac
    done

    set -- "${POSITIONAL[@]}"
}
parse_args "$@"


################################################################################
# Define constants
################################################################################
export FAISS_INDEX_FACTORY="PCAR128,IVF16384,SQfp16"
GPU_TYPE=turing
PARTITION=long
export PROJECT_ROOT="$HOME/ELI5/"

if [[ -z "$USE_PREMAID_EMBS" ]]; then
    export DPR_EMBEDDING_DEPTH=768
    export CREATED_DPR_EMBEDDINGS=true

    # export CREATE_DPR_EMBEDDINGS=true
    export CREATE_DPR_EMBEDDINGS=
    # export CREATE_NP_MEMMAP=true
    export CREATE_NP_MEMMAP=

    LOG_OUT_PATH="$PROJECT_ROOT/logs/out_EMBS_"$FAISS_INDEX_FACTORY"_"$TIMESTAMP".txt"
    LOG_ERR_PATH="$PROJECT_ROOT/logs/err_EMBS_"$FAISS_INDEX_FACTORY"_"$TIMESTAMP".txt"
    JOB_NAME="EMBS_$FAISS_INDEX_FACTORY"
else
    LOG_OUT_PATH="$PROJECT_ROOT/logs/out_"$FAISS_INDEX_FACTORY"_"$TIMESTAMP".txt"
    LOG_ERR_PATH="$PROJECT_ROOT/logs/err_"$FAISS_INDEX_FACTORY"_"$TIMESTAMP".txt"
    JOB_NAME="$FAISS_INDEX_FACTORY"
fi


################################################################################
# Run job(s)
################################################################################
if [[ ! -z "${INTERACTIVE}" ]] ; then
    echo -e "${GREEN}${BOLD}Running in interactive mode${RESET_ALL}"
    source "${PROJECT_ROOT}"/run_baseline_pretrained.sh
else
    echo -e "${GREEN}${BOLD}Running in batch mode${RESET_ALL}"
    TIMESTAMP="$(date +'%Y-%m-%d_%H-%M-%S')"
    sbatch \
    --output="${LOG_OUT_PATH}" \
    --error="${LOG_ERR_PATH}" \
    --partition="${PARTITION}" \
    --export=ALL \
    --mem=128Gb \
    --gres=gpu:"${GPU_TYPE}":1 \
    -J"${JOB_NAME}" \
    --ntasks=1 \
    --time=48:00:00 \
    "${PROJECT_ROOT}"/run_baseline_pretrained.sh
fi
