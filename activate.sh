#!/bin/bash

# Quick activation script for the training environment
# Usage: source activate.sh

if [ ! -f "venv/bin/activate" ]; then
    echo "Error: Virtual environment not found."
    echo "Please run: ./training/setup_env.sh"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Load Kaggle API token if .env exists
if [ -f "training/.env" ]; then
    source training/.env
    export KAGGLE_API_TOKEN
    echo "Environment activated! (Kaggle token loaded)"
else
    echo "Environment activated!"
    echo "Note: training/.env not found. Create it for Kaggle API access."
fi

echo "Python: $(which python)"
echo "Working directory: $(pwd)"
