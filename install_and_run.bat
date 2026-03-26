@echo off
echo [*] ENERGIZING LOCAL SITK BENCH TEST...

:: ELI5: Checking if the "Main Power" (Python) is connected to the building.
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] ERROR: Python not found. Please install Python 3.10+ from python.org
    pause
    exit
)

:: ELI5: Pulling the heavy 240V wire. This downloads the massive PyTorch 
:: libraries directly to the C: drive, so it may take several minutes on the very first run.
echo [*] WIRING THE COMPONENTS (Installing Requirements)...
pip install -r requirements.txt

:: ELI5: Flipping the main breaker to start the production line.
echo [*] FLIPPING THE MAIN BREAKER...

echo [*] WARNING: This script can use a lot of GPU/RAM and may hang a Shadow PC.
echo [*] Choose a mode:
echo    1) Safety check only (no generation)
echo    2) Generate (default, may be heavy)
set /p mode="Enter choice (1/2): "
if "%mode%"=="1" (
    python update_game_ini.py --dry-run
) else (
    python update_game_ini.py
)

echo [!] BENCH TEST COMPLETE.
pause