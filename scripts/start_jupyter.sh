#!/bin/bash
# This script sets up the environment and starts a Jupyter server.
# Designed for use on ETH Zurich's Euler cluster with GPU support.
# You may need to adjust the module load commands based on your environment.

VENV_PATH="$SCRATCH/saliency_analysis/.venv"
export HF_HOME="$SCRATCH/saliency_analysis/cache"

module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

if [ ! -d "$VENV_PATH" ]; then
  RESET_ENV="true"
fi

if [ "$RESET_ENV" == "true" ]; then
  rm -rf "$VENV_PATH"
  python3 -m venv "$VENV_PATH"
  echo "Virtual environment created at $VENV_PATH at $(date)"
fi

source "$VENV_PATH/bin/activate"

if [ "$RESET_ENV" == "true" ]; then
  pip install --quiet --upgrade pip
  pip install --quiet -e .
  pip install --quiet --upgrade -r requirements.txt
  echo "Dependencies installed at $(date)"
else
  echo "Using existing virtual environment at $VENV_PATH"
fi

jupyter notebook --no-browser --ip=0.0.0.0 --port=8888

deactivate