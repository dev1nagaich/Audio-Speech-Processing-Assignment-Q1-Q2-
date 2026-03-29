#!/bin/bash

# Install dependencies for Whisper-small fine-tuning on Hindi conversational speech
# Use this script AFTER creating the conda environment myenv

set -e

echo "=== Installing dependencies for Whisper-small Hindi fine-tuning ==="
echo ""

# Activate conda environment
echo "Activating conda environment 'myenv'..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies from requirements.txt
echo "Installing remaining dependencies..."
pip install \
  transformers>=4.40.0 \
  datasets>=2.18.0 \
  accelerate>=0.29.0 \
  evaluate \
  jiwer \
  librosa \
  soundfile \
  pydub \
  openpyxl \
  pandas \
  numpy \
  tqdm \
  requests \
  indic-transliteration

echo ""
echo "=== Verifying CUDA setup ==="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Number of GPUs: {torch.cuda.device_count()}'); print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')" || echo "Warning: CUDA verification encountered an issue, but installation should be complete."

echo ""
echo "=== Installation complete ==="
echo "Use 'conda activate myenv' before running any scripts."
