#!/usr/bin/env bash
set -euo pipefail

PROMPT=${1:-"Seamless ARK Tek wall, metallic hexagonal panels, glowing blue cyan energy circuits, subtle scratched details"}

python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
pip install -r requirements.txt

python update_game_ini.py --generate-suite --tek-only --width 2048 --height 2048 --prompt "$PROMPT" --save-portfolio