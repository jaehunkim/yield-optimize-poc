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

# Load Kaggle API token from .env if it exists
if [ -f "training/.env" ]; then
    echo "Loading Kaggle API token from .env..."
    source training/.env
elif [ -f ".env" ]; then
    echo "Loading Kaggle API token from .env..."
    source .env
fi

# Check if Kaggle API token is configured
if [ -z "$KAGGLE_API_TOKEN" ] && [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Error: Kaggle API token not found."
    echo ""
    echo "Please set up Kaggle API token:"
    echo "  1. Go to https://www.kaggle.com/account"
    echo "  2. Scroll to 'API Tokens' section"
    echo "  3. Click 'Create New Token'"
    echo "  4. Copy the token (KGAT_...)"
    echo "  5. Create training/.env and add: export KAGGLE_API_TOKEN=your_token"
    echo "  6. Run: source training/.env"
    echo "  7. Run this script again"
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
