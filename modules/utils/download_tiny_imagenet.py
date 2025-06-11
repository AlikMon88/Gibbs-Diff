#!/usr/bin/env python3
"""Download Tiny ImageNet dataset and extract into data directory."""
import os
import requests
from tqdm import tqdm
import zipfile

def download_file(url, dest_path):
    """Download a file from a URL without a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                
if __name__ == '__main__':
    # Ensure dependencies: requests, tqdm
    try:
        import requests, tqdm
    except ImportError:
        print("Please install required packages: pip install requests tqdm")
        exit(1)

    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)

    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_path = os.path.join(data_dir, 'tiny-imagenet-200.zip')

    # Download if not exists
    if not os.path.exists(zip_path):
        download_file(url, zip_path)
    else:
        print(f"{zip_path} already exists, skipping download.")

    # Extract
    extract_dir = os.path.join(data_dir, 'tiny-imagenet-200')
    if not os.path.isdir(extract_dir):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_dir)
        print(f"Extracted to {extract_dir}")
    else:
        print(f"{extract_dir} already exists, skipping extraction.")
