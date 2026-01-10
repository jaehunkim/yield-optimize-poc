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

# Check if Kaggle credentials are configured
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Error: Kaggle API credentials not found."
    echo "Please follow these steps:"
    echo "  1. Go to https://www.kaggle.com/account"
    echo "  2. Click 'Create New API Token'"
    echo "  3. Place kaggle.json in ~/.kaggle/"
    echo "  4. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
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
