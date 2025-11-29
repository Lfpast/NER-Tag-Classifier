#!/usr/bin/env bash
# Setup script for conda environment and project dependencies
# Usage: ./setup.sh [env_name] [python_version] [cpu|gpu] [cuda_version]
# Example (CPU): ./setup.sh ner_env 3.9
# Example (GPU, CUDA 12.6): ./setup.sh ner_env 3.9 gpu 12.6

set -e

ENV_NAME=${1:-ner_env}
PYTHON_VERSION=${2:-3.9}
# Optional mode: cpu (default) or gpu
MODE=${3:-cpu}
# Preferred CUDA version when using GPU. Default to 12.6 for the user's GPU.
CUDA_VERSION=${4:-12.6}

echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -y -n "${ENV_NAME}" python=${PYTHON_VERSION}

echo "Activating environment..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "Installing core packages via conda (pandas, numpy, scikit-learn, nltk, tqdm) ..."
conda install -y -c conda-forge pandas numpy scikit-learn nltk tqdm pip

if [ "${MODE}" = "gpu" ] ; then
    echo "Installing PyTorch with CUDA support (cuda=${CUDA_VERSION}) via conda (pytorch + nvidia channels)..."
    # Note: the package name and channel can change between PyTorch versions. This command uses the pytorch + nvidia channels and the `pytorch-cuda` meta-package.
    # Example command: conda install -y -c pytorch -c nvidia pytorch pytorch-cuda=12.6 torchvision torchaudio
    conda install -y -c pytorch -c nvidia "pytorch" "pytorch-cuda=${CUDA_VERSION}" torchvision torchaudio || {
        echo "GPU install failed. Trying CPU fallback (conda install cpuonly)..."
        conda install -y -c pytorch pytorch torchvision torchaudio cpuonly
    }
else
    echo "Installing PyTorch (CPU-only) via conda..."
    conda install -y -c pytorch pytorch torchvision torchaudio cpuonly
fi

echo "Installing pip packages from requirements.txt (transformers, datasets, evaluate, safetensors, ...)..."
pip install -U pip
pip install -r requirements.txt

echo "Downloading NLTK data (averaged_perceptron_tagger)..."
python - << PYCODE
import nltk
nltk.download('averaged_perceptron_tagger')
try:
    nltk.download('punkt')
except Exception:
    pass
PYCODE

echo "Installation finished. Activate the environment with: conda activate ${ENV_NAME}"

echo "Note: If you are using a GPU, please install a CUDA-enabled PyTorch build by following https://pytorch.org/"

exit 0
