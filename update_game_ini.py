# ELI5: The "Continuity Test." We wrap our imports in a try/except block. 
# If a wire is disconnected (a package didn't install), the script 
# safely stops and tells you exactly which part is missing instead of catching fire.
try:
    import argparse
    import os
    import platform
    import random
    import shutil
    import subprocess
    import sys
    import time
    import torch
    import psutil
    import concurrent.futures
    import threading
    from google.cloud import storage
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler
    from datetime import datetime
    from PIL import Image, ImageDraw
    import pytz
    if platform.system() == "Windows":
        import winsound
except ImportError as e:
    print(f"[!] GROUND FAULT: Missing a critical component -> {e}")
    print("[!] Fix: Double-click 'install_and_run.bat' to wire the panel.")
    exit()


def _get_hf_token():
    """Return HF Hub token from env or None. Silences the unauthenticated-download warning."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def get_runtime_device_and_dtype(device_id=None):
    """Choose the best runtime device/dtype with safe fallback behavior.
    
    Args:
        device_id: Specific CUDA device index (0-N). None means cuda:0.
    """
    if torch.cuda.is_available():
        idx = device_id if device_id is not None else 0
        return torch.device(f"cuda:{idx}"), torch.float16, "fp16"

    # CPU fallback keeps script functional in non-GPU environments.
    return torch.device("cpu"), torch.float32, None


# Default negative prompt to steer SDXL away from common artifacts and illustration-style output.
NEGATIVE_PROMPT = (
    "blurry, low resolution, watermark, text, logo, border, frame, "
    "perspective distortion, fisheye, vignette, photograph, person, "
    "sky, ground, earth, dirt, soil, grass, mud, terrain, horizon, "
    "dry lake bed, cracked earth, salt flat, tessellation, hexagonal tiles, "
    "3D render lighting, shadow banding, "
    "jpeg artifacts, noise, grain, oversaturated, "
    "3D object, miniature, diorama, isometric, item icon, inventory icon, "
    "picture of an object, floating object, product photo, rotated, vertical grain, "
    "cluttered, busy, noisy, ornate, baroque, filigree, painterly, illustration, painting, wall art, mural, story scene, storyboard, 3D scene, camera perspective, depth of field, backlighting, atmosphere, landscape"
    " ,overhead view, birds-eye, close-up, portrait, figure, character, creature"
    " ,abstract, surreal, soft-focus, bokeh, cinematic"
)

# Model presets — add entries here to support --model switching.
MODEL_PRESETS = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
    "playground": "playgroundai/playground-v2.5-1024px-aesthetic",
    "juggernaut": "RunDiffusion/Juggernaut-XL-v9",
}
DEFAULT_MODEL = "sdxl"


# ---------------------------------------------------------------------------
# AI Prompt Generator — mixes styles, materials, palettes, and surface
# treatments to produce fresh SDXL base prompts on each run.
# ---------------------------------------------------------------------------
_AI_STYLES = [
    "cyberpunk megastructure", "brutalist sci-fi", "art-deco starship",
    "alien bio-mechanical hive", "post-apocalyptic military bunker",
    "deep space mining rig", "ancient alien ruins with modern Tek overlay",
    "neon-noir dystopia", "orbital station habitat ring",
    "undersea pressure dome", "volcanic forge outpost",
    "cryo-preservation vault", "quantum research facility",
    "stealth-bomber angular plating", "rusted industrial megafactory",
]
_AI_MATERIALS = [
    "carbon-fiber composite", "brushed titanium", "anodised aluminium",
    "ceramic nano-plating", "ferrocrete with rebar inlay",
    "tungsten-carbide armor", "polished obsidian alloy",
    "layered graphene laminate", "scarred depleted-uranium paneling",
    "corroded copper-verdigris over steel", "frosted quartz composite",
    "matte black stealth coating", "chrome-mirror finish",
]
_AI_PALETTES = [
    "cyan and dark gunmetal", "amber warmth on cold steel",
    "crimson warning stripes on matte black", "teal glow with charcoal",
    "gold filigree on slate grey", "electric violet accents on titanium white",
    "deep ocean blue with rust orange", "neon green traces on dark carbon",
    "warm copper and ice blue", "blood red and brushed silver",
    "holographic iridescent shimmer", "monochrome gunmetal gradient",
]
_AI_SURFACE = [
    "pristine factory-fresh finish with protective film peeling at edges",
    "battle-scarred with plasma burn marks and weld-repair patches",
    "weathered by years of acid rain, micro-pitting across exposed faces",
    "freshly powder-coated with crisp laser-etched serial numbers",
    "cryogenic frost rime on joints with condensation drip trails",
    "heat-discolored from re-entry, rainbow oxide tint near vents",
    "overgrown with bioluminescent fungal tendrils in panel seams",
    "sand-blasted desert wear with fine dust packed into grooves",
    "deep-sea barnacle encrustation with salt corrosion bloom",
    "clean-room sterile, immaculate white with hair-thin panel lines",
]
_AI_LIGHTING = [
    "uniform flat diffuse lighting",
    "subtle emissive glow from recessed channels",
    "cool fluorescent overhead wash",
    "warm forge-light spilling from internal vents",
    "bioluminescent ambient from embedded organisms",
]
_ARCHITECT_INFLUENCES = [
    "Frank Lloyd Wright prairie style, clean horizontal lines",
    "Mayan temple, precise geometric stone blocks",
    "Incan fitted masonry, tight megalithic joints",
    "Tadao Ando, minimalist smooth concrete",
    "Luis Barragán, bold simple color planes",
    "Le Corbusier, raw modular concrete grid",
    "Carlo Scarpa, refined material layering",
    "Zaha Hadid, parametric flowing surfaces",
    "brutalist architecture, massive clean forms",
    "Japanese wabi-sabi, natural material beauty",
    "Art Deco, clean geometric surface pattern",
]

# Interior surface materials per tier — for two-sided building pieces
INTERIOR_MATERIALS = {
    "tek": "clean white composite interior panel, recessed LED channel, smooth matte surface",
    "wood": "smooth whitewashed plaster interior, timber beam shadow, warm tone",
    "stone": "lime-plastered smooth interior wall, subtle stone quoin accent",
    "metal": "painted grey steel interior panel, utility conduit channel, industrial clean",
    "thatch": "woven reed mat interior, bamboo frame shadow, smoke-darkened warm tone",
}

# Pieces that have distinct interior and exterior faces
INTERIOR_PIECES = {
    "wall", "half_wall", "doorframe", "double_doorframe", "window_frame",
    "sloped_wall_left", "sloped_wall_right",
    "door_single", "door_double",
}

# In-memory texture cache — survives across interactive runs for re-rendering
_rendered_textures = {}


def generate_ai_prompt():
    """Produce a base prompt rooted in ARK Ascended vanilla Tek skin (no wall art drift)."""
    return (
        "ARK Ascended vanilla Tek skin reference, dark nanocomposite armor plating, "
        "glowing cyan energy circuit veins, panel seams, industrial welded rivets, "
        "seamless tileable flat material surface texture for walls/floors/ceilings/stairs, "
        "horizontal orientation, minimal artistic decoration, harsh mechanical detail, "
        "UE5 PBR material, no 3D object, no scene, no character, no perspective"
    )


# ---------------------------------------------------------------------------
# Cross-platform timed input (works on Linux pod AND Windows)
# ---------------------------------------------------------------------------
def _timed_input(prompt_text, timeout_seconds=90, default=""):
    """Read a line from stdin with a timeout. Returns *default* on timeout or EOF."""
    result = [default]

    def _reader():
        try:
            result[0] = input(prompt_text)
        except EOFError:
            pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)
    if t.is_alive():
        print(f"\n[*] No input for {timeout_seconds}s — auto-continuing...")
    return result[0]


def print_runtime_diagnostics():
    print(f"[*] PyTorch version: {torch.__version__}")
    print(f"[*] torch.version.cuda: {torch.version.cuda}")
    print(f"[*] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            total_mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"[*] GPU {i}: {gpu_name} ({total_mem_gb:.1f} GB)")
        print(f"[*] Total GPUs: {num_gpus}")

# ELI5: Setting our "Project North." We always use Pacific Time (PST/PDT) 
# in 24-hour military format to keep the blueprints uniform.
def get_pst_time():
    pst = pytz.timezone('America/Los_Angeles')
    return datetime.now(pst).strftime('%Y-%m-%d_%H%M_PST')

# ELI5: The "Main Safety Ground." Per protocol, we MUST back up the 
# server's core power lines to the C: drive before any execution.
def run_ark_backup_protocol(timestamp):
    if os.name == 'nt':
        backup_folder = r'C:\ark_backups'
    else:
        backup_folder = '/root/ark_backups'
    
    # ELI5: "Grading the site." If the backup folder doesn't exist, we build it safely.
    if not os.path.exists(backup_folder):
        try:
            os.makedirs(backup_folder)
            print(f"[*] Built backup conduit: {backup_folder}")
        except Exception as e:
            print(f"[!] GROUND FAULT: Cannot create backup folder on C: drive. Error: {e}")
            return False
    
    files_to_backup = ['Game.ini', 'GameUserSettings.ini', 'update_game_ini.py']
    success = True
    
    for file in files_to_backup:
        if os.path.exists(file):
            try:
                # ELI5: Saving an "As-Built" snapshot of the wiring.
                shutil.copy(file, os.path.join(backup_folder, f"{timestamp}_{file}"))
                print(f"[+] Safety Ground Locked: {file} @ {timestamp}")
            except PermissionError:
                print(f"[!] FILE LOCKED: Cannot back up {file}. Is the ARK server running?")
                success = False
            except Exception as e:
                print(f"[!] Warning: Failed to copy {file}. Error: {e}")
                success = False
        else:
            print(f"[!] Warning: {file} not found in current directory. Skipping.")
    
    return success

# ELI5: The "Load Calculation." Before we order the heavy AI materials, 
# we verify the C: drive has enough square footage (15GB) to hold them.
def set_gpu_power_limit(percentage=87):
    """Cap each GPU's power draw to *percentage* of its default limit via nvidia-smi."""
    if os.name == 'nt' or not torch.cuda.is_available():
        return  # only runs on Linux pods
    try:
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            # Query the default (max) power limit for the card
            result = subprocess.run(
                ["nvidia-smi", "-i", str(i), "--query-gpu=power.default_limit", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            default_watts = float(result.stdout.strip())
            target_watts = int(default_watts * percentage / 100)
            subprocess.run(
                ["nvidia-smi", "-i", str(i), "-pl", str(target_watts)],
                capture_output=True, text=True, timeout=10
            )
            print(f"[GPU {i}] Power limit set to {target_watts}W ({percentage}% of {default_watts:.0f}W)")
    except Exception as e:
        print(f"[!] Could not set GPU power limit: {e}")


def check_system_capacity(allow_cpu=False):
    print("[*] Running Pre-Flight Diagnostics...")
    print_runtime_diagnostics()
    drive_path = 'C:\\' if os.name == 'nt' else '/'
    c_drive = psutil.disk_usage(drive_path)
    free_space_gb = c_drive.free / (1024 ** 3)
    
    if free_space_gb < 15.0:
        print(f"[!] CAPACITY WARNING: Only {free_space_gb:.1f}GB free on {drive_path}.")
        print("[!] The AI model requires ~14GB. Clear space before continuing.")
        return False
    
    # ELI5: The "Multimeter." Checking if the GPU is receiving power.
    if not torch.cuda.is_available():
        if allow_cpu:
            print("[!] WARNING: CUDA GPU not detected. Falling back to CPU inference (slower).")
        else:
            print("[!] CRITICAL ERROR: CUDA GPU not detected. The main breaker is off.")
            return False
        
    print(f"[+] System checks passed. C: Drive Space: {free_space_gb:.1f}GB")
    return True


def make_pipeline_seamless(pipeline):
    """Patches the diffusion pipeline to generate tileable, seamless textures."""
    try:
        for module in pipeline.unet.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.padding_mode = 'circular'
        for module in pipeline.vae.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.padding_mode = 'circular'
        print("[*] Pipeline patched for SEAMLESS TILING (circular padding).")
    except Exception as e:
        print(f"[!] Warning: seamless patch failed: {e}")


def optimize_prompt_load(base_subject, modifiers):
    # CLIP hard limit is 77 tokens. Technical/styled words tokenize at ~2 tokens
    # per word on average, so 35 words is a safe ceiling (35 * 2 = 70 tokens < 77).
    WORD_LIMIT = 40

    combined = f"{base_subject}, {modifiers}".strip(", ") if modifiers else base_subject
    words = combined.split()
    if len(words) <= WORD_LIMIT:
        return combined
    return " ".join(words[:WORD_LIMIT])


def release_pipeline(pipeline):
    # NOTE: Do NOT call pipeline.to("cpu") — diffusers warns that fp16
    # pipelines cannot operate on CPU, flooding the log with false alarms.
    # The pipeline was never intended to run on CPU; we only need to free VRAM.
    try:
        del pipeline
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


# ELI5: The "Stamping Press." This safely loads the AI model into your 
# GPU, complete with circuit breakers in case of a VRAM power surge.
def generate_local_asset(user_prompt, timestamp, width=1024, height=1024):
    print("\n[*] Priming the Shadow PC GPU with safer default output (1024x1024) ...")
    
    try:
        device, torch_dtype, model_variant = get_runtime_device_and_dtype()
        model_id = MODEL_PRESETS.get(os.environ.get("SITK_MODEL", DEFAULT_MODEL), MODEL_PRESETS[DEFAULT_MODEL])
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype,
            variant=model_variant,
            token=_get_hf_token()
        )
        pipe = pipe.to(device)

        make_pipeline_seamless(pipe)

        optimized = optimize_prompt_load(user_prompt, "")
        print(f"[*] Executing high-res render for: '{optimized}' ({width}x{height})")
        seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"[*] Seed: {seed}")
        image = pipe(
            prompt=optimized,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=100,
            guidance_scale=7.5,
            width=width,
            height=height,
            generator=generator,
        ).images[0]
        print("[*] Render pass finished. Please wait while final texture files are being written...")
        
        filename = f"tek_wall_{timestamp}_{width}x{height}.png"

        output_dir = os.path.join(os.getcwd(), "outputs")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, filename)
        image.save(output_path)
        print(f"[+] Asset Successfully Rendered: {output_path}")
        return output_path

    except torch.cuda.OutOfMemoryError:
        print("[!] CIRCUIT OVERLOAD: Out of VRAM. Close background apps (like ARK) and try again.")
    except Exception as e:
        print(f"[!] SYSTEM FAULT: The stamping press jammed: {e}")
    return None


def apply_top_middle_square_cutout(image_path, width, height):
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        # Use actual image dimensions to avoid mismatch with passed arguments.
        img_width, img_height = image.size
        if width != img_width or height != img_height:
            print(f"[!] NOTE: passed dimensions ({width}x{height}) differ from image size ({img_width}x{img_height}), using actual image size")
            width, height = img_width, img_height

        # ARK DevKit standard window placement (based on 300x300 UE unit wall):
        # Window width is 50% (25% margin each side).
        # Window height is 50%, placed 16.6% from the top (50/300 units).
        x1 = int(width * 0.25)
        x2 = int(width * 0.75)
        y1 = int(height * 0.166)
        y2 = int(height * 0.666)

        draw.rectangle((x1, y1, x2, y2), fill=(0, 0, 0))
        image.save(image_path)

        mask = Image.new('L', (width, height), 255)
        mdraw = ImageDraw.Draw(mask)
        mdraw.rectangle((x1, y1, x2, y2), fill=0)
        mask_path = image_path.replace('.png', '_opacity.png')
        mask.save(mask_path)

        print(f"[+] Enforced top-middle square cutout at: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"[+] Opacity mask saved: {mask_path}")
    except Exception as e:
        print(f"[!] Warning: Failed to apply deterministic cutout: {e}")


def generate_uv_wrapped_asset(user_prompt, timestamp, uv_template_path=None, uv_strength=0.65, width=2048, height=2048):
    print("\n[*] UV-AWARE 3D WRAP MODE: starting inpainting/redraw...")

    if uv_template_path and os.path.exists(uv_template_path):
        try:
            uv_image = Image.open(uv_template_path).convert("RGB")
            if uv_image.size != (width, height):
                uv_image = uv_image.resize((width, height), Image.LANCZOS)

            device, torch_dtype, model_variant = get_runtime_device_and_dtype()

            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch_dtype,
                variant=model_variant,
                token=_get_hf_token()
            )
            pipe = pipe.to(device)

            make_pipeline_seamless(pipe)

            local_prompt = (
                f"{user_prompt}, seamless UV texture for ARK, preserve model-proportional openings, "
                "inpaint window and door holes, high-res tileable pattern, no visible seams"
            )

            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(device=device).manual_seed(seed)
            print(f"[*] UV wrap seed: {seed}")
            output_image = pipe(
                prompt=local_prompt,
                image=uv_image,
                strength=uv_strength,
                guidance_scale=7.5,
                num_inference_steps=100,
                generator=generator,
            ).images[0]

            release_pipeline(pipe)

            if not os.path.exists("outputs"):
                os.makedirs("outputs")

            output_path = os.path.join("outputs", f"tek_window_uvwrap_{timestamp}_{width}x{height}.png")
            output_image.save(output_path)
            print(f"[+] UV-wrapped asset saved: {output_path}")
            return output_path

        except Exception as e:
            release_pipeline(pipe)
            print(f"[!] SYSTEM FAULT: UV wrap operation failed: {e}")
            print("[!] Falling back to standard texture generation...")
            return generate_local_asset(user_prompt, timestamp, width, height)
    else:
        print("[!] UV template not provided or not found. Using standard texture generation.")
        return generate_local_asset(user_prompt, timestamp, width, height)


def generate_opacity_mask(piece, width, height):
    mask = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(mask)

    # door/window/gate hole templates in normalized UV space
    if piece == 'window':
        # exact ARK DevKit window placement:
        hole = (int(width * 0.25), int(height * 0.166), int(width * 0.75), int(height * 0.666))
        draw.rectangle(hole, fill=0)
    elif piece == 'door_single':
        hole = (width*0.42, height*0.2, width*0.58, height*0.85)
        draw.rectangle(hole, fill=0)
    elif piece == 'door_tall':
        hole = (width*0.40, height*0.1, width*0.60, height*0.90)
        draw.rectangle(hole, fill=0)
    elif piece == 'door_double':
        hole = (width*0.20, height*0.15, width*0.80, height*0.85)
        draw.rectangle(hole, fill=0)
    elif piece == 'large_gate':
        hole = (width*0.15, height*0.15, width*0.85, height*0.80)
        draw.rectangle(hole, fill=0)
    elif piece == 'behemoth_gate':
        hole = (width*0.10, height*0.10, width*0.90, height*0.90)
        draw.rectangle(hole, fill=0)
    else:
        # default for non-gate pieces; can be full solid or small pattern
        draw.rectangle((0, 0, width, height), outline=255)

    return mask


def get_piece_prompt_theme(piece, tier="tek"):
    tek_themes = {
        'wall': 'carbon-fiber nanocomposite',
        'half_wall': 'polished alloy composite',
        'doorframe': 'machined dark alloy',
        'double_doorframe': 'heavy alloy frame',
        'window_frame': 'precision alloy frame',
        'fence_support': 'hexagonal alloy column',
        'foundation': 'reinforced nanocomposite slab',
        'triangle_foundation': 'nanocomposite slab',
        'pillar': 'translucent energy conduit',
        'cliff_platform_large': 'alloy deck plating',
        'cliff_platform_medium': 'alloy deck plate',
        'cliff_platform_small': 'lightweight alloy tread',
        'tree_platform': 'segmented alloy deck',
        'ocean_platform': 'corrosion-proof alloy',
        'sloped_roof': 'anodised rain-deflector panel',
        'sloped_wall_left': 'angled armor plate',
        'sloped_wall_right': 'angled armor plate',
        'triangle_roof': 'converging shingle panel',
        'roof_intersection': 'valley flashing surface',
        'roof_cap': 'ridge cap alloy',
        'ceiling': 'acoustic dampener panel',
        'triangle_ceiling': 'triangular acoustic panel',
        'hatchframe': 'reinforced hatch frame',
        'trapdoor': 'heavy hatch plate',
        'floor': 'diamond-plate anti-slip',
        'ramp': 'chevron tread alloy',
        'staircase': 'anti-slip tread nosing',
        'spiral_staircase': 'curved alloy tread',
        'fence_foundation': 'narrow composite strip',
        'fence': 'energy mesh field',
        'railing': 'polished alloy rail',
        'door_single': 'armored door slab',
        'door_double': 'split armor panel',
        'dino_gate': 'impact-resistant plating',
        'dino_gateway': 'structural arch frame',
        'behemoth_gate': 'massive armor plate',
        'behemoth_gateway': 'enormous arch buttress',
        'giant_gate': 'ultra-thick composite armor',
        'giant_gateway': 'cathedral-scale arch',
        'ladder': 'anti-slip alloy rung',
        'irrigation_pipe': 'smooth alloy conduit',
    }
    wood_themes = {
        'wall': 'hewn oak plank',
        'half_wall': 'rough-cut plank',
        'doorframe': 'heavy timber frame',
        'double_doorframe': 'massive timber lintel',
        'window_frame': 'hand-carved wood frame',
        'fence_support': 'rough-hewn log post',
        'foundation': 'fieldstone and timber',
        'triangle_foundation': 'fieldstone and log',
        'pillar': 'stripped tree trunk',
        'cliff_platform_large': 'plank deck surface',
        'cliff_platform_medium': 'timber deck plank',
        'cliff_platform_small': 'compact plank deck',
        'tree_platform': 'bark-edge plank deck',
        'ocean_platform': 'tar-sealed plank',
        'sloped_roof': 'cedar shake shingle',
        'sloped_wall_left': 'vertical plank gable',
        'sloped_wall_right': 'vertical plank gable',
        'triangle_roof': 'cedar shake peak',
        'roof_intersection': 'overlapping cedar shakes',
        'roof_cap': 'cedar ridge plank',
        'ceiling': 'rough-sawn rafter',
        'triangle_ceiling': 'angled rafter surface',
        'hatchframe': 'hewn timber hatch frame',
        'trapdoor': 'thick Z-brace plank',
        'floor': 'wide hardwood plank',
        'ramp': 'rough-sawn plank ramp',
        'staircase': 'thick plank tread',
        'spiral_staircase': 'curved plank tread',
        'fence_foundation': 'fieldstone strip',
        'fence': 'rough-split log fence',
        'railing': 'smooth worn log rail',
        'door_single': 'vertical plank door',
        'door_double': 'split plank door',
        'dino_gate': 'heavy log gate',
        'dino_gateway': 'timber arch frame',
        'behemoth_gate': 'massive log gate',
        'behemoth_gateway': 'colossal log arch',
        'giant_gate': 'triple-layer log gate',
        'giant_gateway': 'cathedral timber arch',
        'ladder': 'bark-covered branch',
        'irrigation_pipe': 'hollowed log pipe',
    }
    stone_themes = {
        'wall': 'dressed quarry stone block',
        'half_wall': 'quarried stone course',
        'doorframe': 'hewn stone lintel',
        'double_doorframe': 'massive stone lintel',
        'window_frame': 'carved stone sill',
        'fence_support': 'rough-cut stone post',
        'foundation': 'cyclopean megalith block',
        'triangle_foundation': 'fitted cyclopean stone',
        'pillar': 'drum-stacked stone column',
        'cliff_platform_large': 'fitted stone paver',
        'cliff_platform_medium': 'stone paver surface',
        'cliff_platform_small': 'compact stone slab',
        'tree_platform': 'radial stone paver',
        'ocean_platform': 'salt-weathered stone',
        'sloped_roof': 'overlapping slate tile',
        'sloped_wall_left': 'dressed stone gable',
        'sloped_wall_right': 'dressed stone gable',
        'triangle_roof': 'converging slate tiles',
        'roof_intersection': 'lead-lined slate valley',
        'roof_cap': 'ridge saddle stone',
        'ceiling': 'vaulted stone voussoir',
        'triangle_ceiling': 'triangular vault rib',
        'hatchframe': 'stone frame hatch lip',
        'trapdoor': 'thick slab trapdoor',
        'floor': 'fitted stone flagstone',
        'ramp': 'rough-dressed stone ramp',
        'staircase': 'worn stone tread',
        'spiral_staircase': 'wedge stone tread',
        'fence_foundation': 'narrow rubble-core strip',
        'fence': 'carved stone baluster',
        'railing': 'smooth dressed stone rail',
        'door_single': 'iron-bound stone slab',
        'door_double': 'split stone slab door',
        'dino_gate': 'dressed stone gate panel',
        'dino_gateway': 'carved stone arch',
        'behemoth_gate': 'massive stone gate',
        'behemoth_gateway': 'colossal stone arch',
        'giant_gate': 'titanic stone panel',
        'giant_gateway': 'cathedral stone arch',
        'ladder': 'stone step rungs',
        'irrigation_pipe': 'carved stone aqueduct',
    }
    metal_themes = {
        'wall': 'corrugated industrial steel',
        'half_wall': 'corrugated steel panel',
        'doorframe': 'heavy gauge steel frame',
        'double_doorframe': 'thick steel frame',
        'window_frame': 'rolled steel angle-iron',
        'fence_support': 'square steel tube post',
        'foundation': 'checker-plate steel',
        'triangle_foundation': 'checker-plate steel',
        'pillar': 'cylindrical steel column',
        'cliff_platform_large': 'open-grid steel grating',
        'cliff_platform_medium': 'steel grating deck',
        'cliff_platform_small': 'diamond-tread steel plate',
        'tree_platform': 'segmented steel plate',
        'ocean_platform': 'marine-grade steel',
        'sloped_roof': 'standing-seam metal panel',
        'sloped_wall_left': 'steel panel at pitch',
        'sloped_wall_right': 'steel panel at pitch',
        'triangle_roof': 'standing-seam hip panel',
        'roof_intersection': 'valley gutter steel',
        'roof_cap': 'ridge roll steel cap',
        'ceiling': 'pressed steel ceiling tile',
        'triangle_ceiling': 'triangular pressed steel',
        'hatchframe': 'heavy steel hatch frame',
        'trapdoor': 'thick steel hatch plate',
        'floor': 'diamond-plate steel floor',
        'ramp': 'steel bar-grate ramp',
        'staircase': 'serrated steel tread',
        'spiral_staircase': 'perforated steel tread',
        'fence_foundation': 'narrow steel channel',
        'fence': 'welded steel mesh panel',
        'railing': 'tubular steel rail',
        'door_single': 'reinforced steel door',
        'door_double': 'split steel panel door',
        'dino_gate': 'heavy steel panel gate',
        'dino_gateway': 'structural steel arch',
        'behemoth_gate': 'massive riveted steel',
        'behemoth_gateway': 'enormous steel arch',
        'giant_gate': 'ultra-thick riveted steel',
        'giant_gateway': 'cathedral steel arch',
        'ladder': 'steel knurled rung',
        'irrigation_pipe': 'galvanized steel pipe',
    }
    thatch_themes = {
        'wall': 'woven palm-frond mat',
        'half_wall': 'woven palm mat',
        'doorframe': 'bamboo pole frame',
        'double_doorframe': 'heavy bamboo frame',
        'window_frame': 'split bamboo frame',
        'fence_support': 'whole bamboo pole',
        'foundation': 'packed earth and cobble',
        'triangle_foundation': 'packed earth and cobble',
        'pillar': 'whole bamboo trunk',
        'cliff_platform_large': 'split bamboo deck',
        'cliff_platform_medium': 'bamboo slat deck',
        'cliff_platform_small': 'compact bamboo mat',
        'tree_platform': 'radial bamboo slat',
        'ocean_platform': 'bamboo raft deck',
        'sloped_roof': 'bundled grass thatch',
        'sloped_wall_left': 'woven palm-frond gable',
        'sloped_wall_right': 'woven palm-frond gable',
        'triangle_roof': 'converging thatch bundle',
        'roof_intersection': 'overlapping thatch valley',
        'roof_cap': 'ridge bundle thatch',
        'ceiling': 'woven reed mat ceiling',
        'triangle_ceiling': 'triangular reed mat',
        'hatchframe': 'bamboo hatch frame',
        'trapdoor': 'woven bamboo mat hatch',
        'floor': 'packed earth floor',
        'ramp': 'packed earth ramp',
        'staircase': 'packed earth tread',
        'spiral_staircase': 'bamboo tread platform',
        'fence_foundation': 'packed earth strip',
        'fence': 'vertical bamboo picket',
        'railing': 'horizontal bamboo rail',
        'door_single': 'woven bamboo mat door',
        'door_double': 'split bamboo mat door',
        'dino_gate': 'heavy bundled bamboo gate',
        'dino_gateway': 'bamboo arch frame',
        'behemoth_gate': 'massive lashed bamboo',
        'behemoth_gateway': 'enormous bamboo arch',
        'giant_gate': 'triple-layer bamboo bundle',
        'giant_gateway': 'towering bamboo arch',
        'ladder': 'bamboo rung ladder',
        'irrigation_pipe': 'split bamboo half-pipe',
    }
    all_tier_themes = {
        "tek": tek_themes, "wood": wood_themes, "stone": stone_themes,
        "metal": metal_themes, "thatch": thatch_themes,
    }
    themes = all_tier_themes.get(tier, tek_themes)
    fallback_map = {
        "tek": "carbon-fiber composite surface with cyan plasma traces and hex-bolt grid",
        "wood": "hewn timber plank surface with iron hardware, natural wood grain, horizontal",
        "stone": "dressed quarry stone surface with chisel marks, mortar joints, horizontal courses",
        "metal": "industrial steel surface with rivet rows, weld beads, horizontal orientation",
        "thatch": "woven palm-frond mat surface with bamboo lattice, dried grass fiber, horizontal",
    }
    return themes.get(piece, fallback_map.get(tier, fallback_map["tek"]))


# Serialise model loading so concurrent from_pretrained calls don't
# fight over the same cached safetensors files (causes meta-tensor
# corruption on multi-GPU and Half/Float dtype mismatches).
_model_load_lock = threading.Lock()


def _gpu_worker(gpu_id, jobs, timestamp, uv_template_path, uv_strength, results_lock, generated_paths, model_name="stabilityai/stable-diffusion-xl-base-1.0", num_steps=30, load_barrier=None):
    """Worker that owns one GPU, loads a persistent pipeline, and processes its job queue."""
    torch.cuda.set_device(gpu_id)
    device, torch_dtype, model_variant = get_runtime_device_and_dtype(device_id=gpu_id)
    gpu_label = torch.cuda.get_device_name(gpu_id)
    print(f"[GPU {gpu_id}] Waiting to load pipeline on {gpu_label}...")

    pipe = None
    try:
        with _model_load_lock:
            print(f"[GPU {gpu_id}] Loading {model_name} on {gpu_label}...")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                variant=model_variant,
                token=_get_hf_token()
            ).to(device)
            # Speed: DPM++ 2M Karras scheduler — same quality at 30 steps vs 100 with default
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )
            make_pipeline_seamless(pipe)
            torch.cuda.synchronize(device)
        print(f"[GPU {gpu_id}] Pipeline ready on {gpu_label}. Scheduler: DPM++ 2M Karras, Steps: {num_steps}")
    except Exception as e:
        print(f"[GPU {gpu_id}] FAILED to load pipeline: {e}")
        if load_barrier:
            load_barrier.wait()
        return

    # Wait for ALL GPUs to finish loading before any starts inference.
    # This prevents concurrent .to(device) + inference which triggers
    # CUDA illegal-memory-access on RTX 5090 multi-GPU rigs.
    if load_barrier:
        print(f"[GPU {gpu_id}] Loaded. Waiting for all GPUs before starting inference...")
        load_barrier.wait()
        print(f"[GPU {gpu_id}] All GPUs ready. Starting inference.")

    for job in jobs:
        tier = job["tier"]
        piece = job["piece"]
        spec = job["spec"]
        styled_prompt = job["styled_prompt"]
        width = job["width"]
        height = job["height"]
        side = job.get("side", "ext")

        filename = f"ark_{tier}_{piece}_{side}_{timestamp}_{width}x{height}.png"
        output_path = os.path.join("outputs", filename)

        print(f"[GPU {gpu_id}] Building: {tier} - {piece} -> {width}x{height}")

        if uv_template_path and os.path.exists(uv_template_path):
            wrapped = generate_uv_wrapped_asset(
                styled_prompt, timestamp,
                uv_template_path=uv_template_path,
                uv_strength=uv_strength,
                width=width, height=height
            )
            if wrapped and os.path.exists(wrapped):
                os.replace(wrapped, output_path)
                with results_lock:
                    generated_paths.append(output_path)
                print(f"[GPU {gpu_id}] UV suite asset saved: {output_path}")
                continue

        try:
            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(device=device).manual_seed(seed)
            print(f"[GPU {gpu_id}] {piece} seed: {seed}")

            # Speed: generate at 1024-capped resolution, then upscale with Lanczos
            max_dim = max(width, height)
            if max_dim > 1024:
                scale = max_dim / 1024.0
                gen_w = int((width / scale) // 8 * 8)  # align to 8 for VAE
                gen_h = int((height / scale) // 8 * 8)
                gen_w = max(gen_w, 64)
                gen_h = max(gen_h, 64)
            else:
                gen_w, gen_h = width, height

            image = pipe(
                prompt=styled_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=num_steps,
                guidance_scale=7.5,
                width=gen_w,
                height=gen_h,
                generator=generator,
            ).images[0]

            # Upscale to target resolution if we generated smaller
            if (gen_w, gen_h) != (width, height):
                image = image.resize((width, height), Image.LANCZOS)

            image.save(output_path)
            _rendered_textures[filename] = image.copy()
            with results_lock:
                generated_paths.append(output_path)
            print(f"[GPU {gpu_id}] Suite asset saved: {output_path}")
        except torch.cuda.OutOfMemoryError:
            print(f"[GPU {gpu_id}] CIRCUIT OVERLOAD: VRAM OOM on {piece}. Skipping.")
        except Exception as e:
            print(f"[GPU {gpu_id}] Skipped {tier}/{piece}: {e}")

    release_pipeline(pipe)
    print(f"[GPU {gpu_id}] Worker finished. Pipeline released.")


def generate_ark_building_suite(base_prompt, timestamp, uv_template_path=None, uv_strength=0.65, default_width=2048, default_height=2048, tek_only=False, tiers_filter=None, model_name="stabilityai/stable-diffusion-xl-base-1.0", num_steps=30):
    all_tiers = {
        "tek": "vanilla ARK Ascended Tek structure reference, dark nanocomposite panels, glowing cyan circuitry, industrial armor plating",
        "wood": "primitive wooden structure",
        "stone": "quarried stone structure",
        "metal": "reinforced industrial metal structure",
        "thatch": "primitive thatch and bamboo structure",
    }
    if tiers_filter:
        tiers = {t: all_tiers[t] for t in tiers_filter if t in all_tiers}
    elif tek_only:
        tiers = {"tek": all_tiers["tek"]}
    else:
        tiers = all_tiers

    tek_piece_specs = [
        # --- Walls (6) ---
        {"name": "wall", "prompt": "dark composite panel, clean horizontal seams, faint cyan accent lines", "width": 2048, "height": 2048},
        {"name": "half_wall", "prompt": "dark composite half-panel, polished alloy cap edge, blue indicator strip", "width": 2048, "height": 1024},
        {"name": "doorframe", "prompt": "dark alloy frame surface, clean beveled edges, blue alignment marks", "width": 2048, "height": 2048},
        {"name": "double_doorframe", "prompt": "heavy dark alloy frame, twin track channels, clean machined finish", "width": 2048, "height": 2048},
        {"name": "window_frame", "prompt": "machined dark alloy frame, clean gasket channel, polished bevel", "width": 2048, "height": 2048},
        {"name": "fence_support", "prompt": "hexagonal dark alloy post, vertical conduit line, clean edges", "width": 1024, "height": 2048},
        # --- Foundations & Platforms (8) ---
        {"name": "foundation", "prompt": "dark composite slab, subtle grid pattern, clean matte finish", "width": 2048, "height": 2048},
        {"name": "triangle_foundation", "prompt": "dark composite slab, diagonal structural lines, clean matte", "width": 2048, "height": 2048},
        {"name": "pillar", "prompt": "dark hexagonal column, faint internal cyan glow, clean flanged seams", "width": 1024, "height": 2048},
        {"name": "cliff_platform_large", "prompt": "dark alloy deck plating, clean geometric grip texture, guide markings", "width": 2048, "height": 2048},
        {"name": "cliff_platform_medium", "prompt": "dark carbon-fiber deck, clean edge markers, ventilation slots", "width": 2048, "height": 1536},
        {"name": "cliff_platform_small", "prompt": "dark alloy deck, fine anti-slip texture, clean polished edges", "width": 1536, "height": 1536},
        {"name": "tree_platform", "prompt": "dark alloy radial deck segments, clean expansion joints, matte", "width": 2048, "height": 2048},
        {"name": "ocean_platform", "prompt": "dark corrosion-proof alloy deck, sealed panel seams, clean matte", "width": 2048, "height": 2048},
        # --- Roofs (6) ---
        {"name": "sloped_roof", "prompt": "dark composite shingle panels, clean horizontal overlap rows, matte", "width": 2048, "height": 1536},
        {"name": "sloped_wall_left", "prompt": "angled dark composite panel, clean vent slits, horizontal grain", "width": 2048, "height": 2048},
        {"name": "sloped_wall_right", "prompt": "angled dark composite panel, clean conduit lines, sealed edges", "width": 2048, "height": 2048},
        {"name": "triangle_roof", "prompt": "converging dark alloy panels at ridge, clean apex seam line", "width": 2048, "height": 2048},
        {"name": "roof_intersection", "prompt": "dark composite valley junction, clean drainage channel, sealed", "width": 2048, "height": 2048},
        {"name": "roof_cap", "prompt": "dark alloy ridge cap, horizontal vent strip, clean weather seal", "width": 2048, "height": 1024},
        # --- Ceilings (4) ---
        {"name": "ceiling", "prompt": "dark acoustic panel, recessed LED grid channels, clean flush surface", "width": 2048, "height": 2048},
        {"name": "triangle_ceiling", "prompt": "triangular acoustic panel, subtle LED edge line, clean surface", "width": 2048, "height": 2048},
        {"name": "hatchframe", "prompt": "dark alloy hatch frame lip, clean seal groove, latch recess", "width": 2048, "height": 2048},
        {"name": "trapdoor", "prompt": "dark composite hatch plate, biometric lock ring, anti-slip texture", "width": 2048, "height": 2048},
        # --- Floors & Ramps (4) ---
        {"name": "floor", "prompt": "dark composite anti-slip surface, hex grip zones, subtle joint lines", "width": 2048, "height": 2048},
        {"name": "ramp", "prompt": "dark alloy chevron tread, clean guide strips, magnetic rail groove", "width": 2048, "height": 2048},
        {"name": "staircase", "prompt": "dark composite tread, clean anti-slip nosing, polished alloy accent", "width": 2048, "height": 2048},
        {"name": "spiral_staircase", "prompt": "dark alloy tread, radial anti-slip pattern, central pole ring", "width": 2048, "height": 2048},
        # --- Fences (3) ---
        {"name": "fence_foundation", "prompt": "narrow dark composite strip, anchor nodes, clean slot channel", "width": 2048, "height": 512},
        {"name": "fence", "prompt": "cyan energy mesh between dark alloy nodes, clean grid pattern", "width": 2048, "height": 2048},
        {"name": "railing", "prompt": "polished dark titanium rail, subtle blue light strip on top edge", "width": 2048, "height": 1024},
        # --- Doors (2) ---
        {"name": "door_single", "prompt": "dark composite door slab, biometric panel, clean hinge housing", "width": 2048, "height": 2048},
        {"name": "door_double", "prompt": "dark composite split panels, magnetic seal channel, lock strip", "width": 2048, "height": 2048},
        # --- Gates (6) ---
        {"name": "dino_gate", "prompt": "heavy dark armor plate, clean impact surface, subtle sensor grid", "width": 2048, "height": 2048},
        {"name": "dino_gateway", "prompt": "dark alloy arch frame, force-field track, clean structural ribs", "width": 2048, "height": 2048},
        {"name": "behemoth_gate", "prompt": "massive dark composite armor, hydraulic channels, clean weld seams", "width": 2048, "height": 2048},
        {"name": "behemoth_gateway", "prompt": "enormous dark alloy arch, containment emitters, clean heavy ribs", "width": 2048, "height": 2048},
        {"name": "giant_gate", "prompt": "colossal dark composite armor, energy dampener grid, sealed plates", "width": 2048, "height": 2048},
        {"name": "giant_gateway", "prompt": "cathedral-scale dark alloy arch, beacon strips, structural piers", "width": 2048, "height": 2048},
        # --- Misc (2) ---
        {"name": "ladder", "prompt": "dark alloy anti-slip rung surface, clean bracket mounts", "width": 1024, "height": 2048},
        {"name": "irrigation_pipe", "prompt": "smooth dark alloy pipe, flow-status rings, coupling flanges", "width": 2048, "height": 512},
    ]

    wood_piece_specs = [
        # --- Walls (6) ---
        {"name": "wall", "prompt": "horizontal oak plank surface, iron nail heads, natural wood grain", "width": 2048, "height": 2048},
        {"name": "half_wall", "prompt": "horizontal plank surface, rough-cut top edge, bark remnants", "width": 2048, "height": 1024},
        {"name": "doorframe", "prompt": "heavy timber frame, pegged mortise joints, iron strap marks", "width": 2048, "height": 2048},
        {"name": "double_doorframe", "prompt": "massive timber lintel, axe-cut joints, iron strap bolts", "width": 2048, "height": 2048},
        {"name": "window_frame", "prompt": "carved wood frame, chamfered edges, peg holes, tool marks", "width": 2048, "height": 2048},
        {"name": "fence_support", "prompt": "rough-hewn post, bark remnants, rope lashing grooves", "width": 1024, "height": 2048},
        # --- Foundations & Platforms (8) ---
        {"name": "foundation", "prompt": "fieldstone and timber sill, axe notches, earth chinking", "width": 2048, "height": 2048},
        {"name": "triangle_foundation", "prompt": "fieldstone and log base, iron corner brackets, horizontal", "width": 2048, "height": 2048},
        {"name": "pillar", "prompt": "stripped tree trunk, axe-hewn ends, iron band marks", "width": 1024, "height": 2048},
        {"name": "cliff_platform_large", "prompt": "plank deck, iron bolt heads, rope wear marks, horizontal", "width": 2048, "height": 2048},
        {"name": "cliff_platform_medium", "prompt": "timber deck planks, iron spikes, hemp fiber marks", "width": 2048, "height": 1536},
        {"name": "cliff_platform_small", "prompt": "compact plank deck, iron pins, rope wear groove", "width": 1536, "height": 1536},
        {"name": "tree_platform", "prompt": "bark-edge plank deck, expansion gaps, radial grain", "width": 2048, "height": 2048},
        {"name": "ocean_platform", "prompt": "tar-sealed plank deck, rope grooves, barnacle waterline", "width": 2048, "height": 2048},
        # --- Roofs (6) ---
        {"name": "sloped_roof", "prompt": "cedar shake shingles, horizontal overlapping rows, grey patina", "width": 2048, "height": 1536},
        {"name": "sloped_wall_left", "prompt": "vertical plank gable, horizontal batten shadows, bark trim", "width": 2048, "height": 2048},
        {"name": "sloped_wall_right", "prompt": "vertical plank gable, battens, weathered wood finish", "width": 2048, "height": 2048},
        {"name": "triangle_roof", "prompt": "cedar shake from peak, horizontal overlapping rows, moss", "width": 2048, "height": 2048},
        {"name": "roof_intersection", "prompt": "overlapping cedar shakes at valley, tar sealant, moss", "width": 2048, "height": 2048},
        {"name": "roof_cap", "prompt": "cedar ridge plank, weather boards overlap, tar seam", "width": 2048, "height": 1024},
        # --- Ceilings (4) ---
        {"name": "ceiling", "prompt": "rough-sawn rafter, woven reed mat texture, soot stains", "width": 2048, "height": 2048},
        {"name": "triangle_ceiling", "prompt": "angled rafter, reed mat weave, soot stains, horizontal", "width": 2048, "height": 2048},
        {"name": "hatchframe", "prompt": "hewn timber hatch frame, rawhide hinge marks, latch wear", "width": 2048, "height": 2048},
        {"name": "trapdoor", "prompt": "thick plank, Z-brace pattern, iron ring pull socket", "width": 2048, "height": 2048},
        # --- Floors & Ramps (4) ---
        {"name": "floor", "prompt": "wide hardwood plank, saw marks, peg holes, traffic polish", "width": 2048, "height": 2048},
        {"name": "ramp", "prompt": "rough-sawn plank, cross-slat treads, muddy boot prints", "width": 2048, "height": 2048},
        {"name": "staircase", "prompt": "thick plank tread, worn edges, iron nail heads, horizontal", "width": 2048, "height": 2048},
        {"name": "spiral_staircase", "prompt": "curved plank tread, central pole notches, radial wear", "width": 2048, "height": 2048},
        # --- Fences (3) ---
        {"name": "fence_foundation", "prompt": "low fieldstone strip, packed earth, post sockets, moss", "width": 2048, "height": 512},
        {"name": "fence", "prompt": "rough-split log surface, bark patches, hemp rope fiber", "width": 2048, "height": 2048},
        {"name": "railing", "prompt": "smooth worn log, bark stripped, rope marks, horizontal", "width": 2048, "height": 1024},
        # --- Doors (2) ---
        {"name": "door_single", "prompt": "vertical plank door, Z-brace shadow, iron ring mount", "width": 2048, "height": 2048},
        {"name": "door_double", "prompt": "split plank door, iron butterfly hinges, rope pull wear", "width": 2048, "height": 2048},
        # --- Gates (6) ---
        {"name": "dino_gate", "prompt": "heavy log plank gate, iron band straps, claw scratches", "width": 2048, "height": 2048},
        {"name": "dino_gateway", "prompt": "timber arch frame, notched log joints, iron brackets", "width": 2048, "height": 2048},
        {"name": "behemoth_gate", "prompt": "massive log gate, iron plate overlay, tar coating", "width": 2048, "height": 2048},
        {"name": "behemoth_gateway", "prompt": "colossal log arch, stone pier base, pulley marks", "width": 2048, "height": 2048},
        {"name": "giant_gate", "prompt": "triple-layer log gate, iron cladding, chain-lift marks", "width": 2048, "height": 2048},
        {"name": "giant_gateway", "prompt": "cathedral timber arch, stone buttress, torch scorch", "width": 2048, "height": 2048},
        # --- Misc (2) ---
        {"name": "ladder", "prompt": "bark branch rungs, rope lashing, peg holes, worn grip", "width": 1024, "height": 2048},
        {"name": "irrigation_pipe", "prompt": "bark exterior log pipe, carved coupling joints, tar seal", "width": 2048, "height": 512},
    ]

    stone_piece_specs = [
        # --- Walls (6) ---
        {"name": "wall", "prompt": "dressed quarry stone blocks, chisel marks, clean mortar joints", "width": 2048, "height": 2048},
        {"name": "half_wall", "prompt": "quarried stone courses, rough capstone, clean mortar joints", "width": 2048, "height": 1024},
        {"name": "doorframe", "prompt": "hewn stone lintel frame, mason chisel marks, mortar joints", "width": 2048, "height": 2048},
        {"name": "double_doorframe", "prompt": "massive stone lintel, dressed jamb face, deep mortar joints", "width": 2048, "height": 2048},
        {"name": "window_frame", "prompt": "carved stone sill, mason-tooled chamfer, mortar groove", "width": 2048, "height": 2048},
        {"name": "fence_support", "prompt": "rough-cut stone post, vertical chisel marks, iron anchor", "width": 1024, "height": 2048},
        # --- Foundations & Platforms (8) ---
        {"name": "foundation", "prompt": "cyclopean megalith blocks, fitted stone, thin mortar joints", "width": 2048, "height": 2048},
        {"name": "triangle_foundation", "prompt": "fitted cyclopean stone, mortar joints, diagonal pattern", "width": 2048, "height": 2048},
        {"name": "pillar", "prompt": "drum-stacked stone column, fine-dressed faces, thin mortar", "width": 1024, "height": 2048},
        {"name": "cliff_platform_large", "prompt": "masonry stone platform, dressed ashlar slabs, tight mortar joints, horizontal load-bearing surface", "width": 2048, "height": 2048},
        {"name": "cliff_platform_medium", "prompt": "masonry stone platform, coursed ashlar paving, even horizontal fall-off, mortar joints", "width": 2048, "height": 1536},
        {"name": "cliff_platform_small", "prompt": "masonry stone platform, compact dressed stone tiling, clean joint lines, flat surface", "width": 1536, "height": 1536},
        {"name": "tree_platform", "prompt": "radial stone pavers, mortar in concentric joints", "width": 2048, "height": 2048},
        {"name": "ocean_platform", "prompt": "salt-weathered stone, barnacle crust, eroded mortar", "width": 2048, "height": 2048},
        # --- Roofs (6) ---
        {"name": "sloped_roof", "prompt": "overlapping slate tiles, horizontal rows, natural cleavage", "width": 2048, "height": 1536},
        {"name": "sloped_wall_left", "prompt": "dressed stone gable, horizontal mortar courses, drip mould", "width": 2048, "height": 2048},
        {"name": "sloped_wall_right", "prompt": "dressed stone gable, horizontal courses, vent slit marks", "width": 2048, "height": 2048},
        {"name": "triangle_roof", "prompt": "converging slate rows from apex, overlapping horizontal", "width": 2048, "height": 2048},
        {"name": "roof_intersection", "prompt": "lead-lined slate valley, mortar sealant, moss patches", "width": 2048, "height": 2048},
        {"name": "roof_cap", "prompt": "ridge saddle stones, mortar center line, lichen texture", "width": 2048, "height": 1024},
        # --- Ceilings (4) ---
        {"name": "ceiling", "prompt": "vaulted stone voussoir blocks, mortar joints, soot on keystone", "width": 2048, "height": 2048},
        {"name": "triangle_ceiling", "prompt": "triangular vault, stone ribs at corner, dressed blocks", "width": 2048, "height": 2048},
        {"name": "hatchframe", "prompt": "stone frame lip, iron hinge sockets, mortar, chisel edge", "width": 2048, "height": 2048},
        {"name": "trapdoor", "prompt": "thick stone slab, iron ring-pull socket, peck-dressed face", "width": 2048, "height": 2048},
        # --- Floors & Ramps (4) ---
        {"name": "floor", "prompt": "fitted stone flagstones, tight mortar joints, traffic polish", "width": 2048, "height": 2048},
        {"name": "ramp", "prompt": "rough-dressed stone, horizontal chisel grip lines, drainage", "width": 2048, "height": 2048},
        {"name": "staircase", "prompt": "stone tread, worn bullnose edge, chisel marks, foot polish", "width": 2048, "height": 2048},
        {"name": "spiral_staircase", "prompt": "wedge stone tread, worn pivot point, radiating chisel", "width": 2048, "height": 2048},
        # --- Fences (3) ---
        {"name": "fence_foundation", "prompt": "narrow rubble-core strip, dressed face stones, mortar", "width": 2048, "height": 512},
        {"name": "fence", "prompt": "carved stone baluster, column-slab pattern, mortar joints", "width": 2048, "height": 2048},
        {"name": "railing", "prompt": "smooth dressed stone rail, rounded profile, chisel marks", "width": 2048, "height": 1024},
        # --- Doors (2) ---
        {"name": "door_single", "prompt": "iron-bound stone slab, star bolt heads, pivot socket wear", "width": 2048, "height": 2048},
        {"name": "door_double", "prompt": "split stone slab door, iron center-bar, star bolt pattern", "width": 2048, "height": 2048},
        # --- Gates (6) ---
        {"name": "dino_gate", "prompt": "dressed stone gate panel, iron band grooves, claw marks", "width": 2048, "height": 2048},
        {"name": "dino_gateway", "prompt": "carved stone arch, voussoir keystone, iron bracket holes", "width": 2048, "height": 2048},
        {"name": "behemoth_gate", "prompt": "massive stone gate, iron plate overlay, impact craters", "width": 2048, "height": 2048},
        {"name": "behemoth_gateway", "prompt": "colossal stone arch, buttress ribbing, iron track", "width": 2048, "height": 2048},
        {"name": "giant_gate", "prompt": "titanic stone panel, iron cladding, chain-lift channels", "width": 2048, "height": 2048},
        {"name": "giant_gateway", "prompt": "cathedral stone arch, voussoir blocks, iron straps", "width": 2048, "height": 2048},
        # --- Misc (2) ---
        {"name": "ladder", "prompt": "stone step rungs in wall, chisel-carved, worn grip", "width": 1024, "height": 2048},
        {"name": "irrigation_pipe", "prompt": "carved stone aqueduct, mortar-sealed joints, mineral deposits", "width": 2048, "height": 512},
    ]

    metal_piece_specs = [
        # --- Walls (6) ---
        {"name": "wall", "prompt": "corrugated steel surface, horizontal ribs, rivet rows, rust patina", "width": 2048, "height": 2048},
        {"name": "half_wall", "prompt": "corrugated steel, rolled-edge cap, horizontal ribs, rust streaks", "width": 2048, "height": 1024},
        {"name": "doorframe", "prompt": "heavy gauge steel frame, weld beads, hinge bolt holes", "width": 2048, "height": 2048},
        {"name": "double_doorframe", "prompt": "thick steel frame, twin hinge bolts, roller-track channel", "width": 2048, "height": 2048},
        {"name": "window_frame", "prompt": "rolled steel angle-iron, weld bead corners, bolt holes", "width": 2048, "height": 2048},
        {"name": "fence_support", "prompt": "square steel tube post, welded cap plate, bolt holes", "width": 1024, "height": 2048},
        # --- Foundations & Platforms (8) ---
        {"name": "foundation", "prompt": "heavy checker-plate steel, anchor bolt grid, horizontal welds", "width": 2048, "height": 2048},
        {"name": "triangle_foundation", "prompt": "checker-plate steel on diagonal, hypotenuse weld, bolts", "width": 2048, "height": 2048},
        {"name": "pillar", "prompt": "cylindrical steel column, vertical weld seam, flange bolts", "width": 1024, "height": 2048},
        {"name": "cliff_platform_large", "prompt": "open-grid steel grating, load-bearing bar pattern", "width": 2048, "height": 2048},
        {"name": "cliff_platform_medium", "prompt": "steel grating deck, medium pitch bars, bolt clips", "width": 2048, "height": 1536},
        {"name": "cliff_platform_small", "prompt": "compact diamond-tread steel plate, galvanized finish", "width": 1536, "height": 1536},
        {"name": "tree_platform", "prompt": "segmented steel plates, expansion grooves, radial welds", "width": 2048, "height": 2048},
        {"name": "ocean_platform", "prompt": "marine-grade steel, anti-corrosion coat, salt pitting", "width": 2048, "height": 2048},
        # --- Roofs (6) ---
        {"name": "sloped_roof", "prompt": "standing-seam metal roof panels, horizontal fold seams", "width": 2048, "height": 1536},
        {"name": "sloped_wall_left", "prompt": "steel panel at pitch, horizontal rivet rows, flashed edge", "width": 2048, "height": 2048},
        {"name": "sloped_wall_right", "prompt": "steel panel at pitch, horizontal rivets, louver indents", "width": 2048, "height": 2048},
        {"name": "triangle_roof", "prompt": "converging standing-seam panels at hip, fold seams", "width": 2048, "height": 2048},
        {"name": "roof_intersection", "prompt": "valley gutter steel, welded flashing strip, sealant bead", "width": 2048, "height": 2048},
        {"name": "roof_cap", "prompt": "ridge roll cap steel, folded weather seal, rivet line", "width": 2048, "height": 1024},
        # --- Ceilings (4) ---
        {"name": "ceiling", "prompt": "pressed steel ceiling tiles, horizontal ribs, fixture holes", "width": 2048, "height": 2048},
        {"name": "triangle_ceiling", "prompt": "triangular pressed steel, rib pattern, fixture holes", "width": 2048, "height": 2048},
        {"name": "hatchframe", "prompt": "heavy steel hatch frame, welded latch, hinge recess", "width": 2048, "height": 2048},
        {"name": "trapdoor", "prompt": "thick steel hatch plate, ring-pull recess, diamond-tread", "width": 2048, "height": 2048},
        # --- Floors & Ramps (4) ---
        {"name": "floor", "prompt": "diamond-plate steel floor, raised tread, boot-scuff wear", "width": 2048, "height": 2048},
        {"name": "ramp", "prompt": "steel ramp, horizontal bar-grate tread strips, drainage gaps", "width": 2048, "height": 2048},
        {"name": "staircase", "prompt": "steel tread, serrated anti-slip nosing, bar-grate pattern", "width": 2048, "height": 2048},
        {"name": "spiral_staircase", "prompt": "perforated steel tread, radial anti-slip, galvanized", "width": 2048, "height": 2048},
        # --- Fences (3) ---
        {"name": "fence_foundation", "prompt": "narrow steel channel, bolt-down flange holes, galvanized", "width": 2048, "height": 512},
        {"name": "fence", "prompt": "welded steel mesh panel, diamond grid, angle-iron border", "width": 2048, "height": 2048},
        {"name": "railing", "prompt": "tubular steel rail, brushed grain, weld spots, powder-coat", "width": 2048, "height": 1024},
        # --- Doors (2) ---
        {"name": "door_single", "prompt": "reinforced steel door, pressed rib stiffeners, grey paint", "width": 2048, "height": 2048},
        {"name": "door_double", "prompt": "split steel panel door, central astragal strip, rib pattern", "width": 2048, "height": 2048},
        # --- Gates (6) ---
        {"name": "dino_gate", "prompt": "heavy steel gate panel, welded rib stiffeners, impact dents", "width": 2048, "height": 2048},
        {"name": "dino_gateway", "prompt": "structural steel arch, I-beam flanges, bolt holes", "width": 2048, "height": 2048},
        {"name": "behemoth_gate", "prompt": "massive riveted steel gate, stiffener ribs, hydraulic mounts", "width": 2048, "height": 2048},
        {"name": "behemoth_gateway", "prompt": "enormous steel arch, splice plate rivets, track channels", "width": 2048, "height": 2048},
        {"name": "giant_gate", "prompt": "ultra-thick riveted steel, stiffener grid, chain-lift brackets", "width": 2048, "height": 2048},
        {"name": "giant_gateway", "prompt": "cathedral-scale steel arch, I-beam flanges, splice rivets", "width": 2048, "height": 2048},
        # --- Misc (2) ---
        {"name": "ladder", "prompt": "steel rungs, anti-slip knurled grip, welded rail marks", "width": 1024, "height": 2048},
        {"name": "irrigation_pipe", "prompt": "galvanized steel pipe, coupling threads, valve body", "width": 2048, "height": 512},
    ]

    thatch_piece_specs = [
        # --- Walls (6) ---
        {"name": "wall", "prompt": "woven palm-frond mat, gaps showing dark void behind, bamboo lattice, horizontal", "width": 2048, "height": 2048},
        {"name": "half_wall", "prompt": "woven palm mat, gaps showing dark behind, raw edge, bamboo frame", "width": 2048, "height": 1024},
        {"name": "doorframe", "prompt": "bamboo pole frame, gaps between poles showing dark void, fiber lashing", "width": 2048, "height": 2048},
        {"name": "double_doorframe", "prompt": "heavy bamboo frame, gaps showing dark space, double lashing", "width": 2048, "height": 2048},
        {"name": "window_frame", "prompt": "split bamboo frame, large gaps showing dark void, fiber cord", "width": 2048, "height": 2048},
        {"name": "fence_support", "prompt": "whole bamboo pole, natural nodes, fiber lashing, dark gaps", "width": 1024, "height": 2048},
        # --- Foundations & Platforms (8) ---
        {"name": "foundation", "prompt": "packed earth and cobble, bamboo sill marks, grass fiber", "width": 2048, "height": 2048},
        {"name": "triangle_foundation", "prompt": "packed earth and cobble, bamboo marks, grass fiber", "width": 2048, "height": 2048},
        {"name": "pillar", "prompt": "whole bamboo trunk, natural ring nodes, palm fiber wrapping", "width": 1024, "height": 2048},
        {"name": "cliff_platform_large", "prompt": "split bamboo deck, gaps showing dark void below, fiber lashing", "width": 2048, "height": 2048},
        {"name": "cliff_platform_medium", "prompt": "bamboo slat deck, gaps showing darkness below, fiber lashing", "width": 2048, "height": 1536},
        {"name": "cliff_platform_small", "prompt": "compact bamboo mat, small gaps showing dark space, golden", "width": 1536, "height": 1536},
        {"name": "tree_platform", "prompt": "radial bamboo slats, gaps showing dark void, fiber cord lashing", "width": 2048, "height": 2048},
        {"name": "ocean_platform", "prompt": "bamboo raft deck, gaps showing dark water below, tar seal", "width": 2048, "height": 2048},
        # --- Roofs (6) ---
        {"name": "sloped_roof", "prompt": "grass thatch bundles, thin spots showing dark through, horizontal rows", "width": 2048, "height": 1536},
        {"name": "sloped_wall_left", "prompt": "woven palm-frond gable, gaps showing dark void, horizontal", "width": 2048, "height": 2048},
        {"name": "sloped_wall_right", "prompt": "woven palm-frond gable, gaps showing dark space, horizontal", "width": 2048, "height": 2048},
        {"name": "triangle_roof", "prompt": "converging thatch at peak, thin gaps with dark visible, horizontal", "width": 2048, "height": 2048},
        {"name": "roof_intersection", "prompt": "overlapping thatch at valley, gaps showing dark beneath", "width": 2048, "height": 2048},
        {"name": "roof_cap", "prompt": "ridge bundle thatch, saddle rows, gaps showing dark, cord binding", "width": 2048, "height": 1024},
        # --- Ceilings (4) ---
        {"name": "ceiling", "prompt": "woven reed mat, gaps showing dark rafters behind, smoke stains", "width": 2048, "height": 2048},
        {"name": "triangle_ceiling", "prompt": "triangular reed mat, gaps showing dark rafters, smoke stains", "width": 2048, "height": 2048},
        {"name": "hatchframe", "prompt": "bamboo hatch frame, gaps between poles, dark void, fiber lashing", "width": 2048, "height": 2048},
        {"name": "trapdoor", "prompt": "thick woven bamboo mat, small gaps showing dark below, fiber edge", "width": 2048, "height": 2048},
        # --- Floors & Ramps (4) ---
        {"name": "floor", "prompt": "packed earth floor, grass fiber, foot-traffic polish, flat", "width": 2048, "height": 2048},
        {"name": "ramp", "prompt": "packed earth ramp, bamboo cross-slats, boot prints", "width": 2048, "height": 2048},
        {"name": "staircase", "prompt": "packed earth tread, bamboo nosing, foot-traffic wear", "width": 2048, "height": 2048},
        {"name": "spiral_staircase", "prompt": "bamboo tread, gaps showing dark below, fiber lashing, pole", "width": 2048, "height": 2048},
        # --- Fences (3) ---
        {"name": "fence_foundation", "prompt": "narrow packed earth strip, cobbles, bamboo post sockets", "width": 2048, "height": 512},
        {"name": "fence", "prompt": "vertical bamboo pickets, wide gaps showing dark void behind, reed ties", "width": 2048, "height": 2048},
        {"name": "railing", "prompt": "horizontal bamboo rails, gaps showing dark space, nodes, fiber", "width": 2048, "height": 1024},
        # --- Doors (2) ---
        {"name": "door_single", "prompt": "woven bamboo mat door, gaps showing dark, bamboo frame, fiber", "width": 2048, "height": 2048},
        {"name": "door_double", "prompt": "split bamboo mat, gaps showing dark void, fiber hinges, rope", "width": 2048, "height": 2048},
        # --- Gates (6) ---
        {"name": "dino_gate", "prompt": "heavy bundled bamboo, gaps showing dark behind, fiber bracing", "width": 2048, "height": 2048},
        {"name": "dino_gateway", "prompt": "bamboo arch frame, gaps showing dark void, fiber lashing", "width": 2048, "height": 2048},
        {"name": "behemoth_gate", "prompt": "massive lashed bamboo, gaps showing dark space, vine-rope", "width": 2048, "height": 2048},
        {"name": "behemoth_gateway", "prompt": "enormous bamboo arch, gaps showing dark void, mega lashing", "width": 2048, "height": 2048},
        {"name": "giant_gate", "prompt": "triple-layer bamboo, gaps showing darkness, vine net, lashing", "width": 2048, "height": 2048},
        {"name": "giant_gateway", "prompt": "towering bamboo arch, gaps showing dark void, fiber lashing", "width": 2048, "height": 2048},
        # --- Misc (2) ---
        {"name": "ladder", "prompt": "bamboo rungs, gaps showing dark void, palm fiber lashing, grip", "width": 1024, "height": 2048},
        {"name": "irrigation_pipe", "prompt": "split bamboo half-pipe, fiber cord wraps, mineral stains", "width": 2048, "height": 512},
    ]

    tier_piece_specs = {
        "tek": tek_piece_specs,
        "wood": wood_piece_specs,
        "stone": stone_piece_specs,
        "metal": metal_piece_specs,
        "thatch": thatch_piece_specs,
    }

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    # Build the full job list
    all_jobs = []
    strict_tier_guards = {
        "tek": "vanilla ARK Ascended Tek architecture reference, no wood, no natural stone, no thatch, no ornate art direction, no painterly detail, no perspective, no characters, high-tech metallic surfaces",
        "wood": "authentic wood plank architecture, no stone masonry, no metal panels, no sci-fi elements, natural grain",
        "stone": "masonry stone construction, dressed ashlar stone blocks, precise mortar joints, no wood, no metal panel, no organic cliff textures",
        "metal": "industrial steel construction, welded plates, rivets and beams, no wood, no natural stone, no thatch",
        "thatch": "woven thatch and bamboo construction, natural organic finish, no metal, no stone, no sci-fi",
    }

    for tier, description in tiers.items():
        piece_specs = tier_piece_specs[tier]
        architect = random.choice(_ARCHITECT_INFLUENCES)
        tier_guard = strict_tier_guards.get(tier, "")
        for spec in piece_specs:
            piece = spec["name"]
            width = default_width if spec["width"] == 2048 else min(spec["width"], default_width)
            height = default_height if spec["height"] == 2048 else min(spec["height"], default_height)
            piece_theme = get_piece_prompt_theme(piece, tier=tier)
            # Compose prompt: strictly enforce tier material and piece semantics first.
            base_subject = (
                f"{tier_guard}, {spec['prompt']}, {piece_theme}, {architect}, "
                f"seamless tileable {tier} surface texture, horizontal orientation, "
                f"flat frontal projection, no geometric obstruction, no floating debris, UE5 PBR material"
            )
            full_prompt = base_subject
            styled_prompt = optimize_prompt_load(full_prompt, "")

            all_jobs.append({
                "tier": tier, "piece": piece, "spec": spec,
                "styled_prompt": styled_prompt, "width": width, "height": height,
                "side": "ext",
            })
            # Interior variant for pieces with distinct inside faces
            if piece in INTERIOR_PIECES:
                int_mat = INTERIOR_MATERIALS.get(tier, INTERIOR_MATERIALS["tek"])
                int_subject = (
                    f"{tier_guard}, {int_mat}, {architect}, "
                    f"seamless tileable {tier} interior surface texture, interior architectural detail, "
                    f"horizontal orientation, flat frontal projection, no geometric obstruction, UE5 PBR material"
                )
                int_styled = optimize_prompt_load(int_subject, "")
                all_jobs.append({
                    "tier": tier, "piece": piece, "spec": spec,
                    "styled_prompt": int_styled, "width": width, "height": height,
                    "side": "int",
                })

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"[*] Detected {num_gpus} GPU(s). Distributing {len(all_jobs)} assets across them.")

    # Round-robin distribute jobs to GPUs
    gpu_jobs = [[] for _ in range(num_gpus)]
    for i, job in enumerate(all_jobs):
        gpu_jobs[i % num_gpus].append(job)

    generated_paths = []
    results_lock = threading.Lock()

    if num_gpus == 1:
        # Single GPU: run directly, no thread overhead
        _gpu_worker(0, gpu_jobs[0], timestamp, uv_template_path, uv_strength, results_lock, generated_paths, model_name=model_name, num_steps=num_steps)
    else:
        # Multi-GPU: one thread per GPU, all running in parallel.
        # A barrier ensures ALL pipelines finish loading before ANY begins
        # inference — prevents CUDA illegal-memory-access from overlapping
        # .to(device) transfers with active denoising on other GPUs.
        active_gpus = sum(1 for g in gpu_jobs if g)
        load_barrier = threading.Barrier(active_gpus)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_id in range(num_gpus):
                if gpu_jobs[gpu_id]:
                    futures.append(executor.submit(
                        _gpu_worker, gpu_id, gpu_jobs[gpu_id], timestamp,
                        uv_template_path, uv_strength, results_lock, generated_paths,
                        model_name, num_steps, load_barrier
                    ))
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[!] GPU worker crashed: {e}")

    print(f"[+] Suite complete: {len(generated_paths)} assets generated across {num_gpus} GPU(s).")
    return generated_paths


def upload_outputs_to_bucket(bucket_name, prefix="", local_dir="outputs", session_timestamp=None):
    if not bucket_name:
        print("[!] No bucket name specified for upload. Skipping upload step.")
        return False

    try:
        # ELI5: Use explicit key file when available, env-var fallback for SimplePod.
        key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if key_path and os.path.exists(key_path):
            client = storage.Client.from_service_account_json(key_path)
        else:
            client = storage.Client()
        bucket = client.bucket(bucket_name)

        files_uploaded = []
        for root, _, files in os.walk(local_dir):
            for filename in files:
                if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    continue

                # Only upload files from this session if a timestamp is provided
                if session_timestamp and session_timestamp not in filename:
                    continue

                local_path = os.path.join(root, filename)
                rel_path = os.path.relpath(local_path, local_dir)
                destination = os.path.join(prefix, rel_path).replace('\\', '/')
                blob = bucket.blob(destination)
                blob.upload_from_filename(local_path)
                files_uploaded.append(destination)
                print(f"[+] Uploaded {local_path} to gs://{bucket_name}/{destination}")

        print(f"[+] Upload complete: {len(files_uploaded)} files uploaded to gs://{bucket_name}/{prefix}")
        return True
    except Exception as e:
        print(f"[!] Upload failed: {e}")
        return False


def save_portfolio_description(output_image, uv_template=None):
    try:
        portfolio_text = (
            "# SITK Bench Test Portfolio\n"
            "### Project: ARK Tek Window Texture Pipeline\n"
            "\n"
            "- GPU: NVIDIA CUDA-capable (runtime-detected at execution)\n"
            "- PyTorch: 2.8.0 (CUDA-enabled build)\n"
            "- Model: stabilityai/stable-diffusion-xl-base-1.0\n"
            "- Mode: text2img + UV inpaint support\n"
            "\n"
            f"- Generated texture path: {output_image}\n"
        )
        if uv_template:
            portfolio_text += f"- UV template used: {uv_template}\n"

        with open("PORTFOLIO.md", "w", encoding="utf-8") as fp:
            fp.write(portfolio_text)

        print("[+] PORTFOLIO.md created/updated successfully.")
    except Exception as e:
        print(f"[!] Warning: Could not write portfolio file: {e}")


def resolve_with_holes(args):
    local_pst = pytz.timezone('America/Los_Angeles')

    if args.hole_mode == "mixed":
        return int(datetime.now(local_pst).strftime("%S")) % 2 == 0
    elif args.hole_mode == "with_holes":
        return True
    elif args.hole_mode == "no_holes":
        return False
    else:
        print(f"[!] WARNING: Unknown hole_mode '{args.hole_mode}', defaulting to mixed behavior")
        return int(datetime.now(local_pst).strftime("%S")) % 2 == 0


def validate_args(args):
    if args.images_per_session != 1:
        print(f"[!] images-per-session overridden to 1 (requested {args.images_per_session}).")
        args.images_per_session = 1

    if args.width > args.max_width or args.height > args.max_height:
        print(f"[!] Requested resolution {args.width}x{args.height} exceeds safety max {args.max_width}x{args.max_height}.")
        return False

    if args.uv_template and not os.path.isfile(args.uv_template):
        print(f"[!] WARNING: UV template path not found: {args.uv_template}. Disabling UV wrap.")
        args.uv_template = None

    return True



# ---------------------------------------------------------------------------
# Continuity Buzzer — monitors remote pod output count and beeps on completion.
# Windows-only (winsound). On Linux/Mac, prints bell character instead.
# ---------------------------------------------------------------------------
def start_buzzer_monitoring(target_count, batch_id, node_ip="194.93.48.46", poll_interval=120):
    """Poll the pod for output file count and sound an alarm when target is reached."""
    ssh_key = os.path.expanduser("~/.ssh/id_warsaw_strike")
    remote_cmd = (
        f'ssh -i {ssh_key} -o StrictHostKeyChecking=no root@{node_ip} '
        f'"ls -1 /root/SimplePod_Workspace/outputs/ 2>/dev/null | grep {batch_id} | wc -l"'
    )

    print(f"[*] Continuity Buzzer Energized. Monitoring Batch {batch_id} for {target_count} files...")

    while True:
        try:
            result = subprocess.check_output(remote_cmd, shell=True).decode().strip()
            current_count = int(result)
            timestamp_str = time.strftime("%H:%M:%S")
            print(f"[*] {timestamp_str} - Progress: {current_count}/{target_count} blocks.")

            if current_count >= target_count:
                print(f"[+] TARGET SECURED: {current_count} files detected in output.")
                if platform.system() == "Windows":
                    for _ in range(5):
                        winsound.Beep(1200, 500)
                        time.sleep(0.3)
                else:
                    for _ in range(5):
                        sys.stdout.write("\a")
                        sys.stdout.flush()
                        time.sleep(0.3)
                break
        except Exception as e:
            print(f"[!] Signal Lost: {e}")

        time.sleep(poll_interval)


def main():
    print("--- SITK LOCAL BENCH TEST (INSPECTED & APPROVED) ---")

    # ELI5: Establishing the single "Master Clock" for this session.
    session_timestamp = get_pst_time()

    parser = argparse.ArgumentParser(description="SITK ARK Texture Generator")
    parser.add_argument("--prompt", type=str, default=None, help="Base description for the Tek window wall")
    parser.add_argument("--dry-run", action="store_true", help="Perform backup and checks only; skip generation")
    parser.add_argument("--skip-generation", action="store_true", help="Run backup/checks but skip image generation")
    parser.add_argument("--uv-template", type=str, default=None, help="Path to UV template image for UV-aware wrap")
    parser.add_argument("--uv-strength", type=float, default=0.65, help="Image2Image strength for UV inpainting (0.0-1.0)")
    parser.add_argument("--width", type=int, default=1024, help="Output width in pixels")
    parser.add_argument("--height", type=int, default=1024, help="Output height in pixels")
    parser.add_argument("--max-width", type=int, default=2048, help="Maximum allowed width for this run")
    parser.add_argument("--max-height", type=int, default=2048, help="Maximum allowed height for this run")
    parser.add_argument("--glow", action="store_true", help="Enable glow and emissive lighting modifiers")
    parser.add_argument("--glow-color", type=str, default="cyan", help="Glow color to apply to emissive details")
    parser.add_argument("--generate-suite", action="store_true", help="Generate the full ARK building suite textures")
    parser.add_argument("--tek-only", action="store_true", help="Restrict suite generation to tek tier assets only")
    parser.add_argument("--save-portfolio", action="store_true", help="Write a PORTFOLIO.md summary file")
    parser.add_argument("--hole-mode", choices=["mixed", "with_holes", "no_holes"], default="mixed", help="Control Tek Window Wall opening behavior")
    parser.add_argument("--images-per-session", type=int, default=1, help="Session image cap; enforced to 1")
    parser.add_argument("--upload-bucket", type=str, default="", help="GCS bucket name to upload outputs to (overrides WARSAW_BUCKET)")
    parser.add_argument("--upload-prefix", type=str, default="", help="Optional prefix inside the bucket for uploaded outputs")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face API token (or set HF_TOKEN env var)")
    parser.add_argument("--power-limit", type=int, default=87, help="GPU power cap as percent of TDP (default: 87%%)")
    parser.add_argument("--tiers", type=str, default=None, help="Comma-separated list of tiers to generate (e.g., 'metal,thatch,stone'). Default: all tiers.")
    parser.add_argument("--model", type=str, default=None, choices=list(MODEL_PRESETS.keys()), help="Model preset to use (default: sdxl). Options: " + ", ".join(MODEL_PRESETS.keys()))
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU-only inference when CUDA GPU is not available (slower).")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps per image (default: 30, was 100). Lower = faster.")
    parser.add_argument("--buzzer", type=str, default=None, help="Enable completion buzzer. Format: COUNT:BATCH_ID (e.g., '200:0840')")

    args = parser.parse_args()

    # Step 1: Secure the site
    if not run_ark_backup_protocol(session_timestamp):
        print("[!] Aborting generation to protect ungrounded ARK configurations.")
        return

    # Step 2: Test the voltage and capacity
    if not check_system_capacity(allow_cpu=args.allow_cpu):
        print("[!] Aborting generation due to failed system checks.")
        return

    # Launch buzzer in background thread if requested
    if args.buzzer:
        try:
            parts = args.buzzer.split(":")
            bz_count = int(parts[0])
            bz_batch = parts[1] if len(parts) > 1 else ""
            if not bz_batch:
                raise ValueError("Batch ID required")
            buzzer_thread = threading.Thread(
                target=start_buzzer_monitoring,
                args=(bz_count, bz_batch),
                daemon=True
            )
            buzzer_thread.start()
            print(f"[*] Buzzer armed: will alert when {bz_count} files match \'{bz_batch}\'")
        except (ValueError, IndexError) as e:
            print(f"[!] Invalid --buzzer format. Use COUNT:BATCH_ID (e.g., \'200:0840\'). Error: {e}")

    # Inject HF token into env so _get_hf_token() picks it up everywhere.
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # Set model preset via env so workers pick it up.
    if args.model:
        os.environ["SITK_MODEL"] = args.model

    if not validate_args(args):
        return

    # Cap GPU power draw to protect rented hardware (default 87% of TDP)
    set_gpu_power_limit(args.power_limit)

    if args.dry_run or args.skip_generation:
        base_prompt = ""
    elif args.prompt:
        base_prompt = args.prompt
    else:
        print("\n--- MATERIAL DESIGN INPUT (TEK WINDOW WALL TEST) ---")
        try:
            base_prompt = input("Describe the Tek Window Wall (e.g., 'Seamless ARK Tek window wall, polished obsidian, glowing cyan circuits'): ")
        except EOFError:
            print("[!] Input not available in this environment. Using fallback prompt.")
            base_prompt = "Seamless ARK Tek window wall, polished obsidian, glowing cyan circuits"

    if args.dry_run:
        print("[*] DRY RUN: Backup and checks completed. Generation skipped.")
        return

    if args.skip_generation:
        print("[*] SKIP GENERATION: Backup and checks completed. Exiting.")
        return

    if not base_prompt.strip():
        print("[!] No blueprint provided. Shutting down circuit.")
        return

    texture_modifiers = (
        ", seamless texture, flat frontal projection, albedo diffuse map, "
        "NO borders, NO frames, NO perspective, uniform lighting, UE5 PBR material"
    )

    glow_prompt = ""
    if args.glow:
        glow_prompt = f", glowing {args.glow_color} emissive accents, lit vents, subtle neon highlights"

    with_holes = resolve_with_holes(args)

    hole_directive = (
        ", clear central window cutout, visible open negative space, frame around an actual opening"
        if with_holes
        else ", solid closed wall panel, no cutout, no opening, uninterrupted surface"
    )

    optimized_prompt = (
        "Tek Window Wall test asset, "
        + base_prompt
        + hole_directive
        + texture_modifiers
        + glow_prompt
    )

    print(f"[*] TEST PROFILE: Tek Window Wall | Mode: {'WITH HOLES' if with_holes else 'NO HOLES'} | Images this session: 1")

    # === Generation loop — re-run with new prompts without restarting ===
    run_count = 0
    while True:
        run_count += 1
        if run_count > 1:
            session_timestamp = get_pst_time()

        print(f"\n{'='*60}")
        print(f"[*] RUN #{run_count} | Prompt: {base_prompt[:80]}...")
        print(f"{'='*60}")

        if args.generate_suite:
            # Suite mode: pass only the clean base prompt. The suite function adds
            # per-piece descriptions, themes, and modifiers itself.
            suite_paths = generate_ark_building_suite(
                base_prompt,
                session_timestamp,
                uv_template_path=args.uv_template,
                uv_strength=args.uv_strength,
                default_width=args.width,
                default_height=args.height,
                tek_only=args.tek_only,
                tiers_filter=args.tiers.split(",") if args.tiers else None,
                model_name=MODEL_PRESETS.get(args.model, MODEL_PRESETS[DEFAULT_MODEL]),
                num_steps=args.steps,
            )
            if args.save_portfolio:
                save_portfolio_description("outputs/ark_building_suite", uv_template=args.uv_template)
        elif args.uv_template:
            uv_path = generate_uv_wrapped_asset(optimized_prompt, session_timestamp, uv_template_path=args.uv_template, uv_strength=args.uv_strength, width=args.width, height=args.height)
            if args.save_portfolio:
                if uv_path:
                    save_portfolio_description(uv_path, uv_template=args.uv_template)
                else:
                    save_portfolio_description(f"outputs/tek_window_uvwrap_{session_timestamp}_{args.width}x{args.height}.png", uv_template=args.uv_template)
        else:
            local_path = generate_local_asset(optimized_prompt, session_timestamp, width=args.width, height=args.height)
            if local_path and with_holes:
                apply_top_middle_square_cutout(local_path, args.width, args.height)
            if args.save_portfolio:
                save_portfolio_description(local_path or f"outputs/tek_wall_{session_timestamp}_{args.width}x{args.height}.png")

        # Upload generated images to bucket if requested.
        if args.upload_bucket:
            bucket = args.upload_bucket
        else:
            bucket = os.environ.get("WARSAW_BUCKET", "")

        if bucket:
            preferred_prefix = args.upload_prefix or ""
            upload_outputs_to_bucket(bucket, prefix=preferred_prefix, local_dir="outputs", session_timestamp=session_timestamp)

        print(f"\n{'='*60}")
        print(f"[+] IMAGES RENDERED AND SAVED. Run #{run_count} complete.")
        print(f"{'='*60}")
        print()
        print("  Options:")
        print("    1) Generate MORE with a fresh AI prompt (auto-selected in 90s)")
        print("    2) Enter your own custom prompt")
        print("    3) Re-run with the same prompt")
        print("    4) Quit")
        print()

        choice = _timed_input("Choose [1/2/3/4] (default=1 after 90s): ", timeout_seconds=90, default="1").strip()

        if choice == "4" or choice.lower() in ("quit", "q", "no", "n"):
            print("[*] Session finished. Powering down.")
            break

        if choice == "2":
            new_prompt = _timed_input("Enter new base prompt: ", timeout_seconds=120, default="").strip()
            if new_prompt:
                base_prompt = new_prompt
            else:
                print("[*] Empty input — keeping previous prompt.")

        elif choice in ("1", ""):
            base_prompt = generate_ai_prompt()
            print(f"[AI] New prompt generated:\n      \"{base_prompt}\"")

        # choice == "3" or anything else → keep base_prompt as-is

        # Rebuild optimized_prompt for non-suite single-asset path
        optimized_prompt = (
            "Tek Window Wall test asset, "
            + base_prompt
            + hole_directive
            + texture_modifiers
            + glow_prompt
        )

if __name__ == "__main__":
    main()
