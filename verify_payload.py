import argparse
import os
from google.cloud import storage


def verify_upload(key_path=None, bucket_name='warsawark', prefix='SITK_Deployments/'):
    if key_path is None:
        key_path = r'C:\ark_backups\warsaw-key.json' if os.name == 'nt' else '/root/ark_backups/warsaw-key.json'

    if not os.path.exists(key_path):
        print(f' [!] Missing key file: {key_path}')
        return

    print('[*] Contacting Warsaw Node...')
    try:
        client = storage.Client.from_service_account_json(key_path)
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        
        print(f'[+] Files found in {bucket_name}/{prefix}:')
        count = 0
        for blob in blobs:
            print(f'  - {blob.name} (Size: {blob.size / 1024 / 1024:.2f} MB)')
            count += 1
            
        if count == 0:
            print('  [!] No files found in this directory.')
    except Exception as e:
        print(f' [!] Verification Failed: {e}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verify SITK deployments in Warsaw bucket")
    parser.add_argument("--key-path", default=None, help="Path to service-account JSON key")
    parser.add_argument("--bucket", default="warsawark", help="GCS bucket name")
    parser.add_argument("--prefix", default="SITK_Deployments/", help="GCS prefix to inspect")
    args = parser.parse_args()
    verify_upload(key_path=args.key_path, bucket_name=args.bucket, prefix=args.prefix)
