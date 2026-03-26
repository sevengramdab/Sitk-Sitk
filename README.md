# SITK ARK Texture Suite (Surgical Image-to-Texture Kit)

## Overview
This project generates coherent texture sets for ARK building tiers (thatch, wood, stone, tek) and supports UV-aware inpainting for mesh wrap.

## Architecture
| Component | Role | Stack |
|---|---|---|
| **SITK Strike** (this repo) | The Factory — high-volume asset generation | Python, GCP, 5x RTX 5090 SimplePod |
| **Collision Sphere** (future) | The Sensor — real-time asset detection & UI in UE5 | C++/Blueprints, UE5 |
| **Nitrado Server** | The Job Site — live ARK environment | .ini configs, RCON |

## Requirements
- Python 3.10+
- NVIDIA GPU with CUDA
- PyTorch 2.8.0 + CUDA 12.8 wheel (`cu128`) 
- Model: `stabilityai/stable-diffusion-xl-base-1.0` (full SDXL, not turbo)
- packages in `requirements.txt`

## CUDA Compatibility Notes
- RTX 4090, RTX 4080-class, and future RTX 5090-class GPUs are supported as long as NVIDIA drivers are current and PyTorch is installed from the `cu128` index.
- Runtime device selection is automatic in the generator script (CUDA when available, CPU fallback otherwise).

## Full Pipeline — Windows to SimplePod

### Step 1: Package & Upload from Windows
Place your `warsaw-key.json` in `C:\ark_backups\` then run:
```powershell
powershell -ExecutionPolicy Bypass -File .\deploy_to_warsaw.ps1 -Bucket warsawark
```
This will:
- Package the project (excluding `__pycache__`, `.git`, `.venv`, `outputs`)
- Upload the zip to `gs://warsawark/sitk-bench/`
- Auto-verify the upload landed

### Step 2: Pull & Run on SimplePod (Warsaw Server)
SSH into the pod, then:
```bash
# First time setup — pulls latest zip from GCS and builds venv
cd ~/SimplePod_Workspace
bash setup_pod.sh
```
Or for a quick re-run (venv already exists):
```bash
bash simplepod_run.sh
```

### Step 3: Verify from Windows (optional)
```bash
python verify_payload.py
```

## UV-Aware Mesh Wrapping (for DevKit testing)
Generate textures that respect UV layout for correct hole placement:
```bash
python update_game_ini.py \
  --prompt "seamless tek window wall, frame sill detail" \
  --uv-template "uv_layout.png" \
  --uv-strength 0.65 \
  --width 2048 --height 2048 \
  --save-portfolio
```
Without a UV template, the suite uses deterministic opacity masks for each piece type (window, door_single, door_tall, door_double, large_gate, behemoth_gate).

The suite generates **13 building pieces** per tier across two material tiers:

### Tek Tier
Advanced sci-fi structure — carbon-fiber composite, plasma veins, hex-bolt rivets, ferrocrete, energy conduits.

### Wood Tier
Primitive construction — hand-hewn oak, cedar shake, hemp rope lashing, wrought iron hardware, fieldstone foundations.

### Piece Types (per tier)
wall, roof, fence, foundation, ceiling, floor, window, door_single, door_tall, door_double, hidden_door, large_gate, behemoth_gate

Each piece has tier-specific prompts and style themes. Use `--tek-only` to restrict to tek tier only.

The suite now also applies a different built-in style prompt per part class. Current defaults:
- wall: brushed titanium plates, narrow cyan conduit lanes, modular seam locks
- roof: angled rain-shedding panels, armored ridge caps, faint energy venting
- fence: defensive lattice ribs, reinforced post anchors, exposed power channels
- foundation: heavy structural slabs, recessed service grooves, industrial support ribs
- ceiling: clean recessed panels, maintenance hatches, diffuse light-strip housings
- floor: anti-slip hex tread, segmented plating, durable traffic wear patterns
- window: precision frame trim, integrated glass channel, emissive edge routing
- doors/gates: matching frame, track, hinge, and threshold detail per piece class

## AI Prompt Engine
After each generation run, the interactive loop offers:
1. **Generate more with a fresh AI prompt** (auto-selected after 90 seconds of no input)
2. Enter your own custom prompt
3. Re-run with the same prompt
4. Quit

The AI prompt generator combines from 5 vocabulary pools (~117,000 unique combinations):
- 15 architectural styles
- 13 materials
- 12 colour palettes
- 10 surface treatments
- 5 lighting moods

## Rendering
- **Model:** SDXL Base 1.0 (full model, not turbo — sharper at 2K)
- **Inference steps:** 100 (high detail)
- **Guidance scale:** 7.5
- **Random seed per piece** — logged to console for reproducibility
- **Multi-GPU:** Jobs distributed round-robin across all available GPUs
- **GPU power cap:** 87% of TDP by default (`--power-limit 87`). Protects rented hardware on pod providers. Override with `--power-limit 100` for full throttle.

## Interactive Mode vs Headless
The generator has an interactive loop that lets you pick new prompts between runs. How it works depends on how you launch it:

| Launch method | Behaviour |
|---|---|
| `ssh -t` (interactive TTY) | You see the 4-option menu and can type choices |
| `ssh ... "command"` (no TTY, no pipe) | Menu auto-selects option 1 every 90s (infinite AI prompt loop) |
| `echo 4 \| ssh ... "command"` | Immediately quits after first run ("4" = Quit) |

**For interactive sessions** (you want to see the prompt and choose):
```bash
ssh -t -i ~/.ssh/id_warsaw_strike root@194.93.48.46 \
  "cd /root/SimplePod_Workspace && python3 update_game_ini.py --generate-suite --width 2048 --height 2048 --prompt 'Seamless ARK building texture' --save-portfolio"
```

**For headless continuous generation** (auto-loops with AI prompts):
```bash
ssh -i ~/.ssh/id_warsaw_strike root@194.93.48.46 \
  "cd /root/SimplePod_Workspace && python3 update_game_ini.py --generate-suite --width 2048 --height 2048 --prompt 'Seamless ARK building texture' --save-portfolio --upload-bucket warsawark --upload-prefix SITK_Deployments/outputs"
```

## Quick Commands
```bash
# Standard single texture
python update_game_ini.py --prompt "seamless tek wall" --save-portfolio

# Full tek + wood suite at 2K (26 pieces across 5 GPUs)
python update_game_ini.py --generate-suite --width 2048 --height 2048 \
  --prompt "Seamless ARK building texture" \
  --save-portfolio --power-limit 87

# Tek-only suite (13 pieces)
python update_game_ini.py --generate-suite --tek-only --width 2048 --height 2048 \
  --prompt "Seamless ARK Tek wall, metallic hexagonal panels, glowing blue cyan energy circuits" \
  --save-portfolio

# Upload with custom bucket/prefix
python update_game_ini_gemini.py --bucket warsawark --prefix sitk-bench

# Manual strike (pull latest zip on SimplePod)
python warsaw_cloud_link.py --mode strike
python warsaw_cloud_link.py --mode verify
```

## Output
- Textures saved to `outputs/`
- Portfolio metadata saved to `PORTFOLIO.md`
- Detailed engineering changelog in `DEVLOG.md`

## Key Files
| File | Purpose |
|---|---|
| `update_game_ini.py` | Main generator — text2img + UV wrap + suite mode + AI prompt engine |
| `DEVLOG.md` | Full engineering changelog and run history |
| `update_game_ini_gemini.py` | Package & upload project zip to GCS |
| `warsaw_cloud_link.py` | Pull latest zip from GCS to SimplePod (strike) or verify connection |
| `warsaw_strike.py` | Windows-side strike — backup, zip, upload to `SITK_Deployments/` |
| `verify_payload.py` | List deployed zips in the bucket |
| `deploy_to_warsaw.ps1` | PowerShell wrapper — package, upload, verify in one shot |
| `setup_pod.sh` | SimplePod first-time setup — pull zip, build venv, run suite |
| `simplepod_run.sh` | SimplePod quick re-run — rebuild venv, run suite |

## Changelog
See [DEVLOG.md](DEVLOG.md) for full engineering changelog and run history.# Sitk-Sitk
Sitk Sitk, generates assests on VPNs for Ark Ascended
