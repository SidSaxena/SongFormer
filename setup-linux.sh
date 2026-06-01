#!/bin/bash
set -e

echo "=== SongFormer Setup (Linux + CUDA GPU) ==="

# Create conda environment
conda create -n songformer python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate songformer

# Install PyTorch with CUDA support
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt

# Download model checkpoints
python -c "
import os
os.chdir('src/SongFormer')
from utils.fetch_pretrained import download_all
download_all()
"

echo ""
echo "=== Setup complete! ==="
echo "Run:  conda activate songformer"
echo "Then: python app.py"
echo ""
echo "For a public share link: SONGFORMER_SHARE=1 python app.py"
