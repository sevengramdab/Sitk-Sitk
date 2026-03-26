# SITK Development Log

All engineering changes, bug fixes, and generation runs in reverse chronological order.

---

## 2026-03-26 — Session 4: GPU Power Cap + Interactive Prompt Fix

### GPU Power Limit (87% TDP)
- **Problem:** Rented RTX 5090s on OpenPod should not run at 100% power draw — it's other people's hardware.
- **Fix:** Added `set_gpu_power_limit(percentage)` function. Uses `nvidia-smi -i <gpu> -pl <watts>` to cap each card at 87% of its default TDP (575W × 0.87 = ~500W per card). Runs automatically before generation.
- **CLI:** `--power-limit 87` (default). Override with `--power-limit 100` for full throttle on owned hardware.

### Interactive Prompt Fix (SSH TTY)
- **Problem:** The interactive 4-option menu (AI prompt / custom / reuse / quit) was invisible because `echo 4 |` piped "4" into stdin, auto-selecting Quit after every run. Additionally, non-TTY SSH (`ssh ... "command"`) has no stdin, so `_timed_input()` gets EOF and uses the default.
- **Behaviour matrix:**
  - `ssh -t` (interactive TTY) → user sees menu, can type choices
  - `ssh ... "command"` (no pipe) → auto-defaults to option 1 every 90s = continuous AI prompt loop
  - `echo 4 | ssh ... "command"` → quits after first run
- **Fix:** Documented correct launch methods. For interactive: use `ssh -t` without pipe. For headless continuous: omit `echo 4 |`.

### Generation Run (0804 PST) — Full Success
- **Result:** 26/26 files generated (13 tek + 13 wood), zero errors, no opacity masks
- **GPUs:** 5× RTX 5090 in parallel
- **Model:** SDXL Base 1.0, 100 steps, guidance 7.5, random seed per piece
- **Time:** ~5 minutes total for 26 pieces at 2048×2048
- **Note:** Earlier file count check showed "5" because generation was mid-run — all 26 completed successfully.

---

## 2026-03-26 — Session 3: AI Prompt Engine + Multi-GPU Victory

### Generation Run (0749 PST)
- **Result:** 13/13 pieces generated, 26 files (texture + opacity), zero errors
- **GPUs:** 5× RTX 5090 running in parallel, serialised model loading via `threading.Lock()`
- **Upload:** 26 files → `gs://warsawark/SITK_Deployments/outputs/`

### AI Prompt Generator
Added `generate_ai_prompt()` — a combinatorial prompt engine that randomly mixes:
- 15 architectural styles (cyberpunk megastructure, brutalist sci-fi, orbital station, etc.)
- 13 materials (carbon-fiber composite, polished obsidian alloy, graphene laminate, etc.)
- 12 colour palettes (cyan/gunmetal, amber/steel, holographic iridescent, etc.)
- 10 surface treatments (battle-scarred, cryogenic frost, bioluminescent fungal tendrils, etc.)
- 5 lighting moods

~117,000 unique prompt combinations. No external API dependency — runs offline on the pod.

### Interactive Generation Loop
After each run the operator sees:
```
  1) Generate MORE with a fresh AI prompt (auto-selected in 90s)
  2) Enter your own custom prompt
  3) Re-run with the same prompt
  4) Quit
```
- **90-second auto-continue:** If nobody answers, option 1 fires automatically (new AI prompt, new run).
- Cross-platform `_timed_input()` using a daemon thread — works on Linux pod and Windows.

### Multi-GPU Concurrency Fix
- **Problem:** Concurrent `from_pretrained()` calls from 5 threads fought over cached safetensors files → "Cannot copy out of meta tensor" on GPUs 1–4, dtype corruption (Half/Float mismatch) on GPU 0. Result: 0 assets generated.
- **Fix:** Added `_model_load_lock = threading.Lock()` — each GPU thread acquires the lock before calling `from_pretrained().to(device)`, then releases. Loading is serial; inference is fully parallel.
- **Verified:** All 5 GPUs loaded successfully in sequence, then generated in parallel.

### CLIP 77-Token Overflow Fix
- **Problem:** `optimize_prompt_load()` had WORD_LIMIT = 45. Technical words like "ferrocrete" tokenise at ~2 tokens each → 45 × 2 = 90 tokens, exceeding CLIP's 77-token hard limit.
- **Fix:** Lowered WORD_LIMIT to 35 (35 × 2 = 70 < 77).

### Suite Prompt Bloat Fix
- **Problem:** `main()` was passing `optimized_prompt` (containing "Tek Window Wall test asset" + hole directives + texture modifiers) into `generate_ark_building_suite()`, which added MORE modifiers → contradictory, bloated prompts.
- **Fix:** Pass clean `base_prompt` to the suite. The suite function composes per-piece prompts with its own themes, modifiers, and descriptions.

### Upload Session Filter
- `upload_outputs_to_bucket()` now takes `session_timestamp` and only uploads files containing that timestamp, preventing re-upload of stale outputs.

### Premium Prompt Rewrite
Rewrote all 13 `piece_specs` prompts and all 13 `get_piece_prompt_theme()` entries with detailed, SDXL-optimised descriptions. Added `NEGATIVE_PROMPT` constant applied to all inference calls.

---

## 2026-03-26 — Session 2: CUDA Fix + First Successful Run

### CUDA cu130 → cu128 Fix
- **Problem:** Pod had `torch==2.11.0+cu130` from plain PyPI. NVIDIA driver 575.64.05 supports CUDA 12.9 max — cu130 wheel requires CUDA 13.0 runtime.
- **Fix:** `pip uninstall torch torchvision torchaudio && pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128`
- **Verified:** `torch.cuda.is_available() = True`, `torch.__version__ = 2.8.0+cu128`

### fp16 CPU Warning Elimination
- **Problem:** `release_pipeline()` called `.to("cpu")` on fp16 pipeline → PyTorch warned about fp16 on CPU every time.
- **Fix:** Remove `.to("cpu")` — just `del pipeline` + `torch.cuda.empty_cache()`.

### HF Token Support
- Added `_get_hf_token()` helper reading `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` from env.
- Added `--hf-token` CLI argument.
- Passed `token=_get_hf_token()` to all three `from_pretrained()` calls.

### Multi-GPU Parallel Generation
- Added `_gpu_worker()` function — each thread owns one GPU, loads a pipeline, processes its job queue.
- Jobs distributed round-robin across GPUs.
- `ThreadPoolExecutor(max_workers=num_gpus)` orchestrates the workers.

### First Successful Run (0723 PST)
- 13 tek assets generated serially on GPU 0 (multi-GPU not yet working).
- 26 files uploaded to `gs://warsawark/SITK_Deployments/outputs/`.

---

## 2026-03-26 (Late Night) — Session 1: Full Pipeline Hardening

**All files touched — credentials, CUDA, platform awareness.**

1. **Credential handling unified** — Every file that touches GCS now uses `storage.Client.from_service_account_json()` instead of mutating `os.environ`. Files fixed: `warsaw_strike.py`, `verify_payload.py`, `update_game_ini.py`, `update_game_ini_gemini.py`.
2. **CUDA version mismatch fixed** — `setup_pod.sh` and `simplepod_run.sh` were installing `cu124` wheels while `requirements.txt` specifies `cu128`. Both now use `cu128`.
3. **Platform-aware defaults** — `update_game_ini_gemini.py` backup path, `verify_payload.py` key path, and `deploy_to_warsaw.ps1` all auto-detect Windows vs Linux paths.
4. **deploy_to_warsaw.ps1 upgraded** — Auto-discovers `warsaw-key.json` from `C:\ark_backups\`, runs verify after upload, prints correct SimplePod next-steps.
5. **setup_pod.sh upgraded** — Now calls `warsaw_cloud_link.py --mode strike` to pull the latest zip before building the venv.
6. **UV wrap pipeline leak fixed** — `generate_uv_wrapped_asset()` now calls `release_pipeline()` to free VRAM after each UV wrap pass.
7. **BOM encoding stripped** — `warsaw_strike.py` had a UTF-8 BOM that caused syntax errors on Linux.

---

## 2026-03-26 — Session 0: Warsaw Cloud Link Storage Fixes

**File:** `warsaw_cloud_link.py`

1. **Auto-resolve latest deployment** — `--target-zip` no longer hardcoded to a single timestamped zip. When omitted, `_resolve_latest_zip()` scans `SITK_Deployments/` in the bucket and picks the most recently uploaded `.zip`.
2. **Explicit credential handling** — replaced `os.environ["GOOGLE_APPLICATION_CREDENTIALS"]` with `storage.Client.from_service_account_json()` via shared `_make_storage_client()` helper.
3. **Platform-aware CLI defaults** — `_platform_defaults()` selects Windows or Linux paths automatically.
4. **`verify_warsaw_grid` portability** — `backup_dir` defaults to platform-correct path.
