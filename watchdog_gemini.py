# SAVED FROM GEMINI — NOT ACTIVE. Reference only.
# WARNING: nvidia-smi --gpu-reset is dangerous on shared pods.
# WARNING: pkill -9 python3 kills ALL python processes.
# Consider selective process targeting and softer recovery before using.

import os
import subprocess
import time
import shutil
from datetime import datetime
import pytz

# ELI5: This script is the 'Automatic Transfer Switch' (ATS).
# It monitors the 'Main Breaker' and flips it back on if the circuit trips.

def mandatory_backup():
    workspace = "/root/SimplePod_Workspace"
    backup_dir = "/root/ark_backups"
    os.makedirs(backup_dir, exist_ok=True)

    pst = pytz.timezone('US/Pacific')
    timestamp = datetime.now(pst).strftime('%Y-%m-%d_%H%M_PST')

    configs = ["Game.ini", "GameUserSettings.ini"]
    for cfg in configs:
        src = os.path.join(workspace, cfg)
        if os.path.exists(src):
            dst = os.path.join(backup_dir, f"{timestamp}_{cfg}")
            shutil.copy2(src, dst)
            print(f"[+] Safety Ground Established: {dst}")

def reset_circuitry():
    print("[!] Fault Detected. Flushing 5090 VRAM and killing stalled threads...")
    subprocess.run("pkill -9 python3", shell=True)
    subprocess.run("nvidia-smi --gpu-reset", shell=True)
    time.sleep(5)

def run_watchdog():
    strike_script = "/root/strike_logic.py"
    
    while True:
        check_proc = subprocess.run(f"pgrep -f {strike_script}", shell=True, capture_output=True)
        
        if check_proc.returncode != 0:
            print("[!] Strike Stalled. Initiating Recovery Sequence...")
            mandatory_backup()
            reset_circuitry()
            
            print(f"[*] Re-launching Strike Logic: {strike_script}")
            subprocess.Popen(["python3", strike_script])
        else:
            print("[*] Circuit is Live. 5090s are currently under load.")
            
        time.sleep(60)

if __name__ == "__main__":
    run_watchdog()
