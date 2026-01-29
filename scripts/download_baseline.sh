#!/bin/bash
#
# Download Facebook's pre-trained Croatian FastText vectors for baseline comparison.
# These vectors are trained on Common Crawl + Wikipedia (~6.4B tokens).
#
# Usage:
#   bash scripts/download_baseline.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

# Facebook's pre-trained Croatian vectors URL
URL="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hr.300.bin.gz"
OUTPUT_FILE="$MODELS_DIR/facebook_cc_hr_300.bin"
COMPRESSED_FILE="$MODELS_DIR/cc.hr.300.bin.gz"

echo "=========================================="
echo "Downloading Facebook Croatian FastText"
echo "=========================================="
echo "URL: $URL"
echo "Output: $OUTPUT_FILE"
echo ""

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

# Check if model already exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "Model already exists at: $OUTPUT_FILE"
    echo "Delete the file if you want to re-download."
    exit 0
fi

# Download
echo "Downloading (this may take a while, ~4.2GB compressed)..."
if command -v wget &> /dev/null; then
    wget -c "$URL" -O "$COMPRESSED_FILE"
elif command -v curl &> /dev/null; then
    curl -L -C - "$URL" -o "$COMPRESSED_FILE"
else
    echo "Error: Neither wget nor curl found. Please install one of them."
    exit 1
fi

# Decompress
echo ""
echo "Decompressing..."
gunzip -c "$COMPRESSED_FILE" > "$OUTPUT_FILE"

# Remove compressed file
rm "$COMPRESSED_FILE"

# Verify
if [ -f "$OUTPUT_FILE" ]; then
    SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo ""
    echo "=========================================="
    echo "Download complete!"
    echo "Model saved to: $OUTPUT_FILE"
    echo "Size: $SIZE"
    echo "=========================================="
else
    echo "Error: Download failed"
    exit 1
fi
