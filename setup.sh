#!/usr/bin/env bash
# setup.sh — Install dependencies and download the Korean FastText model.
# Usage: bash setup.sh

set -e

echo "=== Korean Embedding Bias Analysis — Setup ==="

# 1. Install Python dependencies
echo ""
echo "[1/2] Installing Python dependencies into active environment..."
pip install -r requirements.txt

# 2. Download FastText model
echo ""
echo "[2/2] Downloading Korean FastText model (cc.ko.300.bin)..."
echo "      Expected size: ~4.2 GB compressed. This may take 15-60 minutes."
echo "      The download can be safely interrupted and resumed."
echo ""

mkdir -p models

if [ -f "models/cc.ko.300.bin" ]; then
    echo "      cc.ko.300.bin already exists — skipping download."
else
    python -c "
from pathlib import Path
from src.load_embeddings import download_fasttext_korean
download_fasttext_korean(Path('models/'))
"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. (Optional) Place a Korean Word2Vec model at models/ko.bin"
echo "     Download from: https://github.com/Kyubyong/wordvectors"
echo ""
echo "  2. Run the analysis notebook:"
echo "     jupyter notebook notebooks/analysis.ipynb"
echo ""
echo "  3. Or run headless:"
echo "     python run_analysis.py --fasttext-path models/cc.ko.300.bin"
