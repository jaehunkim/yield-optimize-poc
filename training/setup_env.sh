#!/bin/bash

# AdTech POC - Python Environment Setup Script
# Uses pyenv for Python version management

set -e

echo "=== AdTech POC Environment Setup ==="

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "Error: pyenv is not installed. Please install pyenv first."
    echo "Visit: https://github.com/pyenv/pyenv#installation"
    exit 1
fi

# Python version to use
PYTHON_VERSION="3.10.13"

# Check if the Python version is already installed
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
    echo "Installing Python $PYTHON_VERSION via pyenv..."
    pyenv install "$PYTHON_VERSION"
else
    echo "Python $PYTHON_VERSION is already installed."
fi

# Set local Python version for this project
echo "Setting local Python version to $PYTHON_VERSION..."
pyenv local "$PYTHON_VERSION"

# Create virtual environment
VENV_NAME="venv"
echo "Creating virtual environment: $VENV_NAME..."
python -m venv "$VENV_NAME"

# Activate virtual environment
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python packages from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "=== Setup Complete ==="
echo "To activate the virtual environment, run:"
echo "  source training/venv/bin/activate"
echo ""
echo "Note: Make sure to configure your Kaggle API credentials:"
echo "  1. Download kaggle.json from https://www.kaggle.com/account"
echo "  2. Place it in ~/.kaggle/kaggle.json"
echo "  3. chmod 600 ~/.kaggle/kaggle.json"
