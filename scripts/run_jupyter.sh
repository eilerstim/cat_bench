#!/bin/bash
# This script initializes the environment and starts an interactive Jupyter server.
# Designed for use on ETH Zurich's Euler cluster with GPU support.
# You may need to adjust the srun parameters and the `start_jupyter.sh` script based on your environment.

if [ -z "$RESET_ENV" ]; then 
  export RESET_ENV="false" # Set to "true" to reset the virtual environment
fi

srun \
  --gpus=1 \
  --mem-per-cpu=32G \
  --time=02:00:00 \
  --gres=gpumem:64G \
  scripts/start_jupyter.sh