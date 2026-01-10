#!/bin/bash

# AdTech POC - Python Environment Setup Script
# Uses pyenv for Python version management

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== AdTech POC Environment Setup ==="
echo "Project root: $PROJECT_ROOT"
echo "Training directory: $SCRIPT_DIR"

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

# Set local Python version for this project (at root)
echo "Setting local Python version to $PYTHON_VERSION..."
cd "$PROJECT_ROOT"
pyenv local "$PYTHON_VERSION"

# Create virtual environment at project root
VENV_PATH="$PROJECT_ROOT/venv"
echo "Creating virtual environment at: $VENV_PATH..."
python -m venv "$VENV_PATH"

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python packages from requirements.txt..."
cd "$SCRIPT_DIR"
pip install -r requirements.txt

echo ""
echo "=== Setup Complete ==="
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo "Or use the convenience script:"
echo "  source activate.sh"
echo ""
echo "=== Kaggle API Setup ==="
if [ ! -f ".env" ]; then
    echo "Creating .env file for Kaggle API token..."
    echo "Please edit training/.env and add your Kaggle API token."
    echo ""
    echo "Steps to get your token:"
    echo "  1. Go to https://www.kaggle.com/account"
    echo "  2. Scroll to 'API Tokens' section"
    echo "  3. Click 'Create New Token'"
    echo "  4. Copy the token (KGAT_...)"
    echo "  5. Edit training/.env and set KAGGLE_API_TOKEN"
    echo "  6. Load environment: source training/.env"
else
    echo ".env file already exists. Loading Kaggle API token..."
    source .env
    echo "Kaggle API token loaded successfully!"
fi
