#!/bin/bash

# Download iPinYou dataset from Kaggle
# This dataset is already preprocessed by wnzhang/make-ipinyou-data scripts

set -e

echo "=== Downloading iPinYou Dataset from Kaggle ==="

# Check if kaggle CLI is available
if ! command -v kaggle &> /dev/null; then
    echo "Error: kaggle CLI is not installed."
    echo "Please install it: pip install kaggle"
    exit 1
fi

# Get project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check if Kaggle credentials are configured (project .kaggle/ or ~/.kaggle/)
if [ ! -f "$PROJECT_ROOT/.kaggle/kaggle.json" ] && [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Error: Kaggle credentials not found."
    echo ""
    echo "Please set up Kaggle API credentials:"
    echo "  1. Go to https://www.kaggle.com/account"
    echo "  2. Scroll to 'API' section"
    echo "  3. Click 'Create New Token' (downloads kaggle.json)"
    echo "  4. Move kaggle.json to .kaggle/ (in project root)"
    echo "  5. Run: chmod 600 .kaggle/kaggle.json"
    echo "  6. Run this script again"
    exit 1
fi

# Set KAGGLE_CONFIG_DIR to use project .kaggle/ if it exists
if [ -f "$PROJECT_ROOT/.kaggle/kaggle.json" ]; then
    export KAGGLE_CONFIG_DIR="$PROJECT_ROOT/.kaggle"
    echo "Using Kaggle credentials from $KAGGLE_CONFIG_DIR"
fi

# Create data directory
DATA_DIR="training/data/raw"
mkdir -p "$DATA_DIR"

echo "Downloading dataset to $DATA_DIR..."
cd "$DATA_DIR"

# Download iPinYou dataset (6GB)
kaggle datasets download -d lastsummer/ipinyou

# Unzip the dataset
echo "Extracting dataset..."
unzip -q ipinyou.zip

# Remove zip file to save space
rm ipinyou.zip

echo ""
echo "=== Download Complete ==="
echo "Dataset location: $DATA_DIR"
echo ""
echo "Next steps:"
echo "  1. Explore the dataset structure"
echo "  2. Run load_data.py to prepare data for training"
