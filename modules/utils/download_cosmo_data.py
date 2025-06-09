#!/usr/bin/env python3
"""Download .fits files from the remote cosmo data directory into data/cosmo."""
import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def fetch_html(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def download_file(url, dest_path):
    """Download a file from a URL with a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(dest_path)}",
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

def main():
    # Ensure dependencies: requests, tqdm, beautifulsoup4
    try:
        import requests, tqdm, bs4
    except ImportError:
        print("Please install required packages: pip install requests tqdm beautifulsoup4")
        exit(1)

    base_url = 'https://users.flatironinstitute.org/~bburkhart/data/CATS/MHD/256/b.1p.01/'
    data_dir = os.path.join(os.getcwd(), 'data', 'cosmo')
    os.makedirs(data_dir, exist_ok=True)

    print(f"Fetching base directory listing from {base_url}")
    html = fetch_html(base_url)
    soup = BeautifulSoup(html, 'html.parser')
    # Find subdirectories (hrefs ending with '/'), excluding parent link
    subdirs = [
        a['href'] for a in soup.find_all('a', href=True)
        if a['href'].endswith('/') and a['href'] != '../'
    ]

    for sub in subdirs:
        sub_url = urljoin(base_url, sub)
        print(f"Processing subdirectory: {sub_url}")
        sub_html = fetch_html(sub_url)
        soup2 = BeautifulSoup(sub_html, 'html.parser')
        # Find .fits files
        fits_links = [
            a['href'] for a in soup2.find_all('a', href=True)
            if a['href'].lower().endswith('.fits')
        ]
        if not fits_links:
            print(f"No .fits files found in {sub_url}")
            continue

        # Create local subdirectory
        subdir_local = os.path.join(data_dir, sub.rstrip('/'))
        os.makedirs(subdir_local, exist_ok=True)

        for href in fits_links:
            file_url = urljoin(sub_url, href)
            dest_path = os.path.join(subdir_local, href)
            if os.path.exists(dest_path):
                print(f"{dest_path} exists, skipping.")
            else:
                download_file(file_url, dest_path)

if __name__ == '__main__':
    main()
