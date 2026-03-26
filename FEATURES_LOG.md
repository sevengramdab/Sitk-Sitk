# SITK ARK Texture Pipeline — Feature Development Log

> **Project**: Surgical Image-to-Texture Kit (SITK)  
> **Purpose**: Automated texture generation for ARK: Survival Evolved building pieces  
> **Infrastructure**: 5× NVIDIA RTX 5090 (32GB each) on SimplePod → GCS → Shadow PC  
> **Stack**: Python 3.10, PyTorch 2.8.0+cu128, Stable Diffusion XL Base 1.0, diffusers 0.37.1

---

## Feature 1: Multi-GPU Parallel Generation Engine

**Problem**: Generating 200+ high-res textures (2048×2048) on a single GPU takes 17+ hours.

**Solution**: Built a thread-per-GPU worker pool that distributes texture jobs across all available GPUs using round-robin scheduling.

**How it works**:
- On startup, `torch.cuda.device_count()` discovers all GPUs
- A `threading.Lock` serializes model loading (prevents VRAM race conditions during `from_pretrained()`)
- Each GPU thread loads its own persistent SDXL pipeline and processes its job queue independently
- `concurrent.futures.ThreadPoolExecutor` manages the thread pool
- OOM errors are caught per-piece (skips that piece, continues the queue)

**Result**: 5 GPUs process 205 textures in ~40 minutes vs ~17 hours on a single card. True 5× throughput scaling.

---

## Feature 2: 5-Tier Building Material System (205 Textures)

**Problem**: ARK has 5 building tiers (tek, wood, stone, metal, thatch) × 41 unique building pieces each. Every texture needs tier-appropriate surface material descriptions.

**Solution**: Hand-authored surface-material prompt dictionaries for all 5 tiers with 40 unique piece themes each, plus per-tier piece spec generators.

**How it works**:
- `get_piece_prompt_theme()` maps every piece name to a tier-specific flat surface material description
- Tek: sci-fi materials (carbon-fiber, titanium alloy, plasma conduits, nano-composite plating)
- Wood: natural materials (hewn oak, cedar shake, iron hardware, rope lashing)
- Stone: masonry (quarried limestone, chisel marks, mortar joints, slate tiles)
- Metal: industrial (corrugated steel, rivets, welds, checker-plate, grating)
- Thatch: primitive (bamboo, palm fronds, woven reeds, packed earth)
- `_build_tier_piece_specs()` generates piece specs with correct resolutions per building piece

**Key constraint**: All prompts describe FLAT SURFACE MATERIALS only — never the 3D object itself. This prevents SDXL from rendering miniature pictures of railings/walls instead of tileable material textures.

**41 pieces per tier**: wall, half_wall, doorframe, double_doorframe, window_frame, fence_support, foundation, triangle_foundation, pillar, cliff_platform_large/medium/small, tree_platform, ocean_platform, sloped_roof, sloped_wall_left/right, triangle_roof, roof_intersection, roof_cap, ceiling, triangle_ceiling, hatchframe, trapdoor, floor, ramp, staircase, spiral_staircase, fence_foundation, fence, railing, door_single, door_double, dino_gate, dino_gateway, behemoth_gate, behemoth_gateway, giant_gate, giant_gateway, ladder, irrigation_pipe + extras.

---

## Feature 3: Seamless Tileable Texture System

**Problem**: Standard diffusion outputs have visible seams when tiled as game textures.

**Solution**: Patched the SDXL pipeline's convolution layers to use circular (toroidal) padding.

**How it works**:
- `make_pipeline_seamless()` iterates every `torch.nn.Conv2d` in both UNet and VAE
- Sets `padding_mode = 'circular'` on each layer
- This wraps edge pixels to the opposite side during convolution, making left↔right and top↔bottom edges match seamlessly
- Applied to every pipeline before any inference call

**Result**: All generated textures tile seamlessly with zero visible seams — ready for direct UE5 import.

---

## Feature 4: Model Switching System

**Problem**: Different SDXL model checkpoints have different strengths. Need to A/B test quality across models without code changes.

**Solution**: CLI-driven model preset system with environment variable propagation to GPU workers.

**How it works**:
- `MODEL_PRESETS` dict maps friendly names to HuggingFace model paths:
  - `sdxl` → stabilityai/stable-diffusion-xl-base-1.0 (default, highest quality)
  - `sdxl-turbo` → stabilityai/sdxl-turbo (4-step fast generation)
  - `playground` → playgroundai/playground-v2.5-1024px-aesthetic
  - `juggernaut` → RunDiffusion/Juggernaut-XL-v9
- `--model sdxl-turbo` sets `os.environ["SITK_MODEL"]` which all GPU workers read
- Resolved to full HuggingFace path before passing to `generate_ark_building_suite()`

---

## Feature 5: AI Prompt Generator

**Problem**: Manually writing 200+ unique prompts per run is impractical. Repeated prompts produce visual monotony.

**Solution**: Combinatorial vocabulary mixer that randomly selects from 5 style pools per generation.

**Pools**:
| Pool | Count | Example |
|------|-------|---------|
| Styles | 15 | "cyberpunk megastructure", "volcanic forge outpost" |
| Materials | 12 | "brushed titanium", "frosted quartz composite" |
| Colour Palettes | 12 | "cyan and dark gunmetal", "neon green on dark carbon" |
| Surface Treatments | 10 | "battle-scarred with plasma burns", "cryogenic frost rime" |
| Lighting Moods | 5 | "uniform flat diffuse", "warm forge-light" |

**Result**: ~97,200 unique prompt combinations. Each run produces visually distinct textures.

---

## Feature 6: CLIP Token Overflow Protection

**Problem**: SDXL's CLIP text encoder has a hard 77-token limit. Longer prompts get silently truncated, losing important details at the end.

**Solution**: Front-load the most important material description, truncate to 40 words max.

**How it works**:
- `optimize_prompt_load()` splits the prompt, keeps only the first 40 words (~77 CLIP tokens)
- Prompt composition puts material description FIRST, framing/modifiers LAST
- Critical terms like "seamless tileable flat material surface texture" lead the prompt
- Decorative terms that can be safely truncated go at the end

---

## Feature 7: GPU Power Management

**Problem**: Running 5× RTX 5090s at full TDP risks thermal throttling and hardware damage on rented pod hardware.

**Solution**: Automated nvidia-smi power capping at configurable percentage (default 87%).

**How it works**:
- `set_gpu_power_limit()` queries each GPU's default TDP via `nvidia-smi`
- Calculates target watts: `default_limit * (percentage / 100)`
- Sets power cap per GPU with `nvidia-smi -i {gpu_id} -pl {watts}`
- Logs each GPU's cap for transparency (e.g., "GPU 0: 500W (87% of 575W)")

**Result**: 5 GPUs run at 500W each instead of 575W, preventing thermal issues while maintaining 90%+ generation speed.

---

## Feature 8: Remote Completion Buzzer

**Problem**: Generation runs take 30-40 minutes. No way to know when it's done without checking manually.

**Solution**: Background SSH polling thread that monitors output file count and beeps on completion.

**How it works**:
- `--buzzer "200:0955"` launches a daemon thread
- Thread SSHes into pod every 120 seconds, runs `ls | grep BATCH_ID | wc -l`
- When count ≥ target: Windows plays 5× 1200Hz beeps via `winsound.Beep()`, Linux sends `\a` bell chars
- Non-blocking: runs alongside generation or standalone from local machine

---

## Feature 9: Live Progress Dashboard (`progress.ps1`)

**Problem**: Need real-time visibility into generation progress across all 5 GPUs.

**Solution**: PowerShell monitoring script with visual progress bar, ETA calculation, and live GPU telemetry.

**Dashboard displays**:
- Block-character progress bar (█░) with count/total and percentage
- Time-based ETA: `(elapsed / count) × remaining = seconds left`
- Per-GPU metrics via `nvidia-smi`: utilization %, VRAM used/total, temperature, power draw
- Updates every 30 seconds, beeps on completion

---

## Feature 10: Interactive Re-Prompt Loop

**Problem**: Want to iterate on texture prompts without restarting the full pipeline (model reload, backup, checks).

**Solution**: Post-generation interactive menu with 90-second auto-continue.

**Options after each run**:
1. **Generate more with AI prompt** — auto-selected after 90s timeout
2. **Custom prompt** — manual input with 120s timeout
3. **Re-run same prompt** — repeat with new random seed
4. **Quit**

**Key detail**: Pipeline stays loaded in VRAM between runs. Only the prompt changes. New PST timestamp per run for unique filenames.

---

## Feature 11: UV-Aware Mesh Wrapping (Inpaint Mode)

**Problem**: Some building pieces need openings (windows, doors) placed exactly where the 3D mesh expects them.

**Solution**: Image-to-Image inpainting mode that generates texture onto a UV template.

**How it works**:
- `--uv-template uv_layout.png` activates inpaint mode
- `generate_uv_wrapped_asset()` uses SDXL img2img pipeline
- `--uv-strength 0.65` controls how much the AI can deviate from the template
- `generate_opacity_mask()` creates deterministic masks per piece type (window, door, gate)
- Mask coordinates use UV-normalized proportions for consistent placement

---

## Feature 12: GCS Cloud Upload Pipeline

**Problem**: Textures generate on a rented pod in Poland. Need to get them to local machine reliably.

**Solution**: Session-filtered Google Cloud Storage upload with automatic credential discovery.

**How it works**:
- `upload_outputs_to_bucket()` scans the outputs folder
- Only uploads files matching the current session timestamp (avoids re-uploading old runs)
- Authenticates via `GOOGLE_APPLICATION_CREDENTIALS` or explicit key path
- Supports `--upload-bucket` and `--upload-prefix` CLI overrides
- File types: .png, .jpg, .jpeg, .webp only

---

## Feature 13: Pre-Flight Safety Systems

**Problem**: Silent failures waste expensive GPU time. Bad configs corrupt game files.

**Solution**: Multi-layer safety checks before any generation begins.

| Check | What it does |
|-------|-------------|
| **INI Backup** | Copies Game.ini and GameUserSettings.ini to timestamped backup folder |
| **Disk Space** | Validates 15GB+ free before starting |
| **GPU Presence** | Confirms CUDA available with PyTorch |
| **Resolution Limits** | `--max-width`/`--max-height` ceiling prevents OOM crashes |
| **VRAM Cleanup** | `release_pipeline()` avoids fp16-to-CPU warning spam |
| **OOM Recovery** | Per-piece catch in GPU workers — logs error, skips piece, continues queue |

---

## Feature 14: Negative Prompt Engineering

**Problem**: SDXL generates miniature 3D objects instead of flat surface materials when given piece names like "railing" or "window_frame."

**Solution**: Aggressive negative prompt targeting the specific failure modes.

**NEGATIVE_PROMPT includes**:
> 3D object, miniature, diorama, isometric, item icon, inventory icon, picture of an object, floating object, product photo, rotated, vertical grain, blurry, watermark, text, logo, border, frame, perspective distortion

Combined with base prompt starting "seamless tileable flat material surface texture, NOT a 3D object" — forces SDXL to produce flat material textures.

---

## Infrastructure Summary

| Component | Spec |
|-----------|------|
| **Compute** | 5× NVIDIA RTX 5090 (32GB GDDR7 each) on SimplePod |
| **Local** | Shadow PC — AMD EPYC 3.7GHz 8vCores, 1× RTX A4500 20GB |
| **Model** | Stable Diffusion XL Base 1.0 (100 steps, guidance 7.5) |
| **Resolution** | 2048×2048 (walls), 2048×1024 (half-walls), 1024×2048 (pillars), adaptive for platforms |
| **Output** | 205 seamless tileable PNG textures per full run |
| **Runtime** | ~40 minutes for full 5-tier suite on 5× RTX 5090 |
| **Pipeline** | Python → SSH → SimplePod → GCS → Shadow PC |
| **Transport** | SCP for code deploy, GCS for bulk texture output |
