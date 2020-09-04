#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/home/mila/g/gagnonju/logs/faiss_dpr.txt
#SBATCH --error=/home/mila/g/gagnonju/logs/faiss_dpr.txt
#SBATCH --ntasks=1
#SBATCH --time=48:00
#SBATCH --gres=gpu:turing:1
#SBATCH --partition=unkillable


###############################################################################
# Utilities
###############################################################################
# Directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Different terminal styling constants
GREEN="\e[32m"
RED="\033[31m"
BLUE="\e[34m"
BOLD="\e[1m"
RESET_ALL="\e[0m"
RESET_FG="\e[39m"

# Debugging logging function that shows which python executable we are using
function python_exec {
    echo -e "Current python executable: $BOLD$BLUE$(which python)$RESET_ALL"
}

# Test if we are on SLURM. Not super pretty. WIP.
# Useful for things we only want to do when doing a non-interactive job
case "$-" in
    *i*)	NOT_INTERACTIVE=false ;;
    *)  	NOT_INTERACTIVE=true ;;
esac

################################################################################
# Main Action
################################################################################
if $NOT_INTERACTIVE; then
    echo ""
    echo "###########################################################"
    echo "# Node information:"
    echo "###########################################################"

    echo -e "Hostname: $BLUE$BOLD$HOSTNAME$RESET_ALL"
    GPU_INFO="$(nvidia-smi --query-gpu=gpu_name,memory.total --format=csv | tail -n 1)"
    echo -e "GPU info: $BLUE$BOLD$GPU_INFO$RESET_ALL"
fi


echo ""
echo "###########################################################"
echo "# Installing Python"
echo "###########################################################"

module purge
module refresh
# The following makes conda available in our env
module load anaconda/3
source $CONDA_ACTIVATE

echo ""
echo "###########################################################"
echo "# Load Modules"
echo "###########################################################"

echo -e "Loading the Pytorch module"
module load pytorch

echo -e "Loading the Cuda modules"
module load cuda/10.0
module load cuda/10.0/cudnn/7.6

# Detect if we've already built the virtual environment.
if [[ "$(conda env list | grep -Eo ^\\w+ | grep -E eli5)" != "" ]] ; then 
    # We have. Activate the virtual environment.
    echo ""
    echo "###########################################################"
    echo "# Activating the VENV"
    echo "###########################################################"
    
    echo -e "Activating venv ..."
    conda activate eli5
    echo "Done activating venv."
else
    # We haven't. Create the virtual environment.
    echo ""
    echo "###########################################################"
    echo "# ${BLUE}Building${RESET_FG} and activating the VENV"
    echo "###########################################################"

    echo -e "\nCreating venv eli5"
    yes | conda create -n eli5
    echo "Done creating venv eli5."

    echo -e "\nActivating venv ..."
    conda activate eli5
    echo "Done activating venv."
    
    echo "\nInstalling packages with conda ..."
    PACKAGES=($(cat requirements.txt | sed -z 's/\n/ /g'))
    conda config --env --append channels conda-forge
    conda install "${PACKAGES[@]}" -y
    conda install faiss-gpu cudatoolkit=10.1 -c pytorch -y
    echo "Done installing packages with conda."

    echo "Installing packages with pip ..."
    python -m pip install pyarrow==0.16 nlp colored-traceback absl-py\
        matplotlib git+https://github.com/huggingface/transformers.git
    echo "Done installing packages with pip."
fi

# Required for Pytorch
export LD_LIBRARY_PATH="/cvmfs/ai.mila.quebec/apps/x86_64/debian/openmpi/2.1.6/lib/:$LD_LIBRARY_PATH"


################################################################################
# Checks
################################################################################
echo ""
echo "###########################################################"
echo "# Checks"
echo "###########################################################"
CHECK="$BOLD[$GREEN✓$RESET_FG]$RESET_ALL -"
X_MARK="$BOLD[$RED✗$RESET_FG]$RESET_ALL -"

# Check if we have the correct python executable
CORRECT_PYTHON=/home/mila/g/gagnonju/.conda/envs/eli5/bin/python
if [[ "$(which python)" != "$CORRECT_PYTHON" ]] ; then
    echo -e "$X_MARK Incorrect Python executable:"
    echo -e "\tGot:     \t$(which python)"
    echo -e "\tExpected:\t$CORRECT_PYTHON"
    exit 1
else
    echo -e "$CHECK Correct Python executable."
fi

# Check if we can import FAISS
python -c "import colored_traceback.auto; import faiss"
if [[ $? -eq 1 ]] ; then
    echo -e "$X_MARK Could not import FAISS."
    exit 1
else
    echo -e "$CHECK Imported FAISS successfully."
fi

# Check if we can import Pytorch
python -c "import colored_traceback.auto; import torch"
if [[ $? -eq 1 ]] ; then
    echo -e "$X_MARK Could not import Pytorch."
    exit 1
else
    echo -e "$CHECK Imported Pytorch successfully."
fi
