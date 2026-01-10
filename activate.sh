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

# Check if packages are installed, install if not
if ! python -c "import kaggle" 2>/dev/null; then
    echo "Installing Python packages..."
    pip install -q -r training/requirements.txt
fi

# Check Kaggle credentials
if [ -f .kaggle/kaggle.json ]; then
    export KAGGLE_CONFIG_DIR=.kaggle
    echo "Environment activated! (Kaggle credentials found in .kaggle/)"
elif [ -f ~/.kaggle/kaggle.json ]; then
    echo "Environment activated! (Kaggle credentials found in ~/.kaggle/)"
else
    echo "Environment activated!"
    echo "Note: Kaggle credentials not found. Add kaggle.json to .kaggle/ for Kaggle API access."
fi

echo "Python: $(which python)"
echo "Working directory: $(pwd)"
