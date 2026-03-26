import argparse
import os
import shutil
import zipfile
from datetime import datetime
import zoneinfo
import pytz
from google.cloud import storage


def _make_storage_client(key_path):
    """Create a storage client explicitly from a service-account JSON key."""
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"Missing key file: {key_path}")
    return storage.Client.from_service_account_json(key_path)


def _resolve_latest_zip(client, bucket_name, prefix="SITK_Deployments/"):
    """Find the most recently uploaded SITK package in the bucket."""
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    zips = [b for b in blobs if b.name.endswith(".zip")]
    if not zips:
        raise FileNotFoundError(f"No .zip files found under gs://{bucket_name}/{prefix}")

    packaged_zips = [b for b in zips if os.path.basename(b.name).startswith("sitk_bench_")]
    candidates = packaged_zips or zips
    latest = max(candidates, key=lambda b: b.updated)
    print(f"[*] Auto-resolved latest deployment: {latest.name}")
    return latest.name


# ELI5: This is the 'Main Breaker' for the Surgical Strike.
# We use this to pull the SITK 'Block' and explode it into our workspace.
def execute_strike(
    workspace="/root/SimplePod_Workspace",
    backup_dir="/root/ark_backups",
    key_path="/root/warsaw-key.json",
    bucket_name="warsawark",
    target_zip=None,
):
    os.makedirs(workspace, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)

    try:
        client = _make_storage_client(key_path)
    except FileNotFoundError as e:
        print(f"[!] {e}")
        return False

    # --- MANDATORY BACKUP PROTOCOL ---
    pst = pytz.timezone("US/Pacific")
    timestamp = datetime.now(pst).strftime("%Y-%m-%d_%H%M_PST")

    configs = ["Game.ini", "GameUserSettings.ini"]
    for cfg in configs:
        cfg_path = os.path.join(workspace, cfg)
        if os.path.exists(cfg_path):
            backup_path = os.path.join(backup_dir, f"{timestamp}_{cfg}")
            shutil.copy2(cfg_path, backup_path)
            print(f"[+] Safety Ground Established: {backup_path}")

    # --- ASSET RETRIEVAL ---
    try:
        if target_zip is None:
            target_zip = _resolve_latest_zip(client, bucket_name)

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(target_zip)
        local_zip = os.path.join(workspace, "payload.zip")
        print("[*] Extracting Block from Warsaw feed...")
        blob.download_to_filename(local_zip)

        print(f"[*] Unpacking assets into {workspace}...")
        with zipfile.ZipFile(local_zip, "r") as zip_ref:
            zip_ref.extractall(workspace)

        os.remove(local_zip)
        print("[+] SUCCESS: Node Primed. 5x RTX 5090 Circuitry energized for strike.")
        return True
    except Exception as e:
        print(f"[!] Strike Aborted: {e}")
        return False


def verify_warsaw_grid(backup_dir=None, json_key_path=None):
    la_tz = zoneinfo.ZoneInfo("America/Los_Angeles")
    timestamp = datetime.now(la_tz).strftime("%Y-%m-%d_%H%M_PST")

    if backup_dir is None:
        backup_dir = r"C:\ark_backups" if os.name == "nt" else "/root/ark_backups"

    if json_key_path is None:
        json_key_path = os.path.join(backup_dir, "warsaw-key.json")

    game_ini = r"C:\ShooterGame\Saved\Config\WindowsServer\Game.ini"
    gus_ini = r"C:\ShooterGame\Saved\Config\WindowsServer\GameUserSettings.ini"

    print(f"[*] Initiating PST Backup Circuit: {timestamp}")

    os.makedirs(backup_dir, exist_ok=True)

    for file in [game_ini, gus_ini]:
        if os.path.exists(file):
            shutil.copy2(file, os.path.join(backup_dir, f"{timestamp}_{os.path.basename(file)}"))
            print(f" [+] Redundancy copy created: {os.path.basename(file)}")
        else:
            print(f" [!] Circuit Open: {os.path.basename(file)} not detected at local path.")

    print(f"[*] Validating Security Badge from: {json_key_path}")

    try:
        storage_client = _make_storage_client(json_key_path)
    except FileNotFoundError:
        print(" [!] CRITICAL FAULT: warsaw-key.json missing from local junction box.")
        return False

    print("[*] Attempting to handshake with Google Cloud Utility Grid...")

    try:
        buckets = list(storage_client.list_buckets())
        print("[*] Connection Terminated Successfully (Loop Closed)!")
        print(f" [+] Found {len(buckets)} active storage units on the grid:")
        for bucket in buckets:
            print(f"  - {bucket.name}")
        return True
    except Exception as e:
        print(f" [!] Ground Fault Error: {e}")
        return False


def _platform_defaults():
    """Return (workspace, backup_dir, key_path) appropriate for the current OS."""
    if os.name == "nt":
        return (r"C:\SimplePod_Workspace", r"C:\ark_backups", r"C:\ark_backups\warsaw-key.json")
    return ("/root/SimplePod_Workspace", "/root/ark_backups", "/root/warsaw-key.json")


if __name__ == "__main__":
    _ws, _bak, _key = _platform_defaults()

    parser = argparse.ArgumentParser(description="Warsaw Cloud Link: verify or strike")
    parser.add_argument("--mode", choices=["verify", "strike"], default="strike", help="Operation mode")
    parser.add_argument("--workspace", default=_ws)
    parser.add_argument("--backup-dir", default=_bak)
    parser.add_argument("--key-path", default=_key)
    parser.add_argument("--bucket", default="warsawark")
    parser.add_argument("--target-zip", default=None,
                        help="Blob path to a specific zip; omit to auto-resolve the latest deployment")
    args = parser.parse_args()

    if args.mode == "verify":
        passed = verify_warsaw_grid(backup_dir=args.backup_dir, json_key_path=args.key_path)
    else:
        passed = execute_strike(
            workspace=args.workspace,
            backup_dir=args.backup_dir,
            key_path=args.key_path,
            bucket_name=args.bucket,
            target_zip=args.target_zip,
        )

    if not passed:
        raise SystemExit(1)
