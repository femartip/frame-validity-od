import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path

JSON_FILE = 'data/STF/data_links.json'  
OUTPUT_DIR = 'data/STF/' 
MAX_WORKERS = 8  

def download_entry(entry):
    try:
        rel_path = entry['key']
        download_url = entry['url']
        
        local_filepath = Path(OUTPUT_DIR) / rel_path
        
        if rel_path.endswith('/'):
            local_filepath.mkdir(parents=True, exist_ok=True)
            return 
        
        local_filepath.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(download_url, stream=True)
        response.raise_for_status() # Check for HTTP errors


        with open(local_filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
    except Exception as e:
        print(f"Error downloading {entry.get('key')}: {e}")

def main():
    if not os.path.exists(JSON_FILE):
        print(f"Error: {JSON_FILE} not found.")
        return

    print("Loading JSON...")
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    entries = data.get('urls', [])
    print(f"Found {len(entries)} items to process.")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(download_entry, entries), total=len(entries), unit="file"))

    print(f"\nDownload complete! Files are in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()