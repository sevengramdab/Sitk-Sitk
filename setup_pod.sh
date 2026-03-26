#!/usr/bin/env bash
set -euo pipefail

cd ~/SimplePod_Workspace

# Pull the latest SITK package from Warsaw bucket
python warsaw_cloud_link.py --mode strike

# Build clean venv
python -m venv .venv
source .venv/bin/activate

# Install cu128 PyTorch FIRST (matches requirements.txt)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -q

# Verify CUDA
python -c "import torch; print('[CUDA]', torch.__version__, torch.cuda.is_available())"

# Install remaining deps from requirements.txt (skip torch lines)
grep -vE '^torch|^torchvision|^torchaudio|--extra-index-url|--index-url' requirements.txt | pip install -r /dev/stdin -q

echo "[+] Environment ready. Starting generation..."
python update_game_ini.py \
  --generate-suite --tek-only \
  --width 2048 --height 2048 \
  --prompt "Seamless ARK Tek wall, metallic hexagonal panels, glowing blue cyan energy circuits, subtle scratched details" \
  --save-portfolio
