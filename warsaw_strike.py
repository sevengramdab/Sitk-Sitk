import os
import shutil
from datetime import datetime
import zoneinfo
from google.cloud import storage


def _make_storage_client(key_path):
    """Create a storage client explicitly from a service-account JSON key."""
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"Missing key file: {key_path}")
    return storage.Client.from_service_account_json(key_path)


def find_ark_paths():
    common_paths = [
        r"C:\Program Files (x86)\Steam\steamapps\common\ARK Survival Ascended\ShooterGame\Saved\Config\WindowsServer",
        r"C:\XboxGames\ARK- Survival Ascended\Content\ShooterGame\Saved\Config\WindowsServer",
        r"C:\ShooterGame\Saved\Config\WindowsServer"
    ]
    for path in common_paths:
        if os.path.exists(path):
            return os.path.join(path, "Game.ini"), os.path.join(path, "GameUserSettings.ini")
    return None, None

def execute_strike():
    la_tz = zoneinfo.ZoneInfo("America/Los_Angeles")
    timestamp = datetime.now(la_tz).strftime('%Y-%m-%d_%H%M_PST')
    
    backup_dir = r"C:\ark_backups"
    json_key_path = os.path.join(backup_dir, "warsaw-key.json")
    
    game_ini, gus_ini = find_ark_paths()
    
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    print(f"[*] Starting SITK Surgical Strike: {timestamp}")

    if game_ini and gus_ini:
        for file in [game_ini, gus_ini]:
            if os.path.exists(file):
                shutil.copy2(file, os.path.join(backup_dir, f"{timestamp}_{os.path.basename(file)}"))
                print(f" [+] Backup Locked: {os.path.basename(file)}")
    else:
        print(" [!] ARK config paths not found dynamically. Skipping backup.")

    try:
        storage_client = _make_storage_client(json_key_path)
    except FileNotFoundError as e:
        print(f" [!] CRITICAL FAULT: {e}")
        return

    sitk_source = r"C:\SITK_Bench_Test" 
    zip_output = r"C:\ark_backups\SITK_Archive"
    
    print("[*] Compressing SITK assets...")
    shutil.make_archive(zip_output, 'zip', sitk_source)
    zip_filename = f"{zip_output}.zip"
    
    print("[*] Uploading Payload to warsawark...")
    try:
        bucket = storage_client.bucket("warsawark")
        blob = bucket.blob(f"SITK_Deployments/{timestamp}_SITK_Archive.zip")
        
        blob.upload_from_filename(zip_filename)
        print(f" [+] Payload Delivered: {blob.name}")
        
        os.remove(zip_filename)
        print(" [+] Local ZIP wiped. Strike complete.")
        
    except Exception as e:
        print(f" [!] Payload Transfer Failed: {e}")

if __name__ == '__main__':
    execute_strike()
