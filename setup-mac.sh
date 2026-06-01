#!/bin/bash
set -e

echo "=== SongFormer Setup (Mac — CPU/MPS) ==="
echo ""
echo "Note: Mac does not support NVIDIA GPUs."
echo "On Apple Silicon (M1/M2/M3/M4), MPS acceleration is automatic."
echo "On Intel Mac, inference runs on CPU (slower)."
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH."
    echo ""
    echo "Fix: Install Miniconda from https://docs.conda.io/en/latest/miniconda.html"
    echo "Then restart your terminal and retry."
    exit 1
fi

# Create conda environment
conda create -n songformer python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate songformer

# Install PyTorch (CPU-only — no CUDA on Mac)
pip install torch==2.8.0 torchaudio==2.8.0

# Install remaining dependencies (mac-compatible, excludes pesq and gpustat)
pip install -r requirements-mac.txt

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
echo "On Apple Silicon, MPS acceleration is automatic."
echo "For a public share link: SONGFORMER_SHARE=1 python app.py"
