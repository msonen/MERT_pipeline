import os
import mirdata
import requests
from tqdm import tqdm

DATA_ROOT = "./raw_data"

def is_folder_populated(folder_path):
    """Checks if a folder exists and is not empty."""
    if os.path.exists(folder_path) and len(os.listdir(folder_path)) > 0:
        return True
    return False

def download_file(url, dest_folder, filename):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    filepath = os.path.join(dest_folder, filename)
    # Double check specifically for the zip file
    if os.path.exists(filepath):
        print(f"Skipping {filename} (file already exists)")
        return

    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def download_all():
    # Debug: Print available datasets to confirm names
    # print("Available mirdata datasets:", mirdata.list_datasets())

    # 1. GiantSteps (Key Detection)
    print("\n--- Checking GiantSteps (Key) ---")
    gs_path = f"{DATA_ROOT}/giantsteps"
    if is_folder_populated(gs_path):
        print(f"Data present in {gs_path}. Skipping download.")
    else:
        gs_url = "https://github.com/GiantSteps/giantsteps-key-dataset/archive/refs/heads/master.zip"
        download_file(gs_url, gs_path, "giantsteps_key.zip")
    
    # 2. VocalSet (Singer ID)
    print("\n--- Checking VocalSet ---")
    vs_path = f"{DATA_ROOT}/vocalset"
    if is_folder_populated(vs_path):
        print(f"Data present in {vs_path}. Skipping download.")
    else:
        vs_url = "https://zenodo.org/record/1193957/files/VocalSet.zip?download=1"
        download_file(vs_url, vs_path, "VocalSet.zip")

    # 3. GTZAN (Genre) - CORRECTED IDENTIFIER
    print("\n--- Checking GTZAN ---")
    gtzan_path = f"{DATA_ROOT}/gtzan"
    
    if is_folder_populated(gtzan_path):
        print(f"Data present in {gtzan_path}. Skipping mirdata check/download.")
    else:
        try:
            # Use 'gtzan_genre' instead of 'gtzan'
            gtzan = mirdata.initialize('gtzan_genre', data_home=gtzan_path)
            gtzan.download() 
            print("GTZAN download initiated via mirdata.")
        except ValueError:
            print("Error: 'gtzan_genre' not found. Printing available datasets:")
            print(mirdata.list_datasets())
        except Exception as e:
            print(f"GTZAN Auto-download failed: {e}")
            print("Please download manually from Kaggle: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification")

    # 4. Instructions for the Hard Ones
    print("\n" + "="*50)
    print("MANUAL DOWNLOAD REQUIRED FOR RESTRICTED DATASETS")
    print("="*50)
    print(f"1. EmoMusic: Go to https://cvml.unige.ch/databases/emoMusic/ -> Save to {DATA_ROOT}/emomusic")
    print(f"2. MagnaTagATune: Go to http://mirg.city.ac.uk/ -> Save to {DATA_ROOT}/mtt")
    print(f"3. MTG-Jamendo: Run the following command manually:")
    print(f"   git clone https://github.com/MTG/mtg-jamendo-dataset {DATA_ROOT}/mtg")

if __name__ == "__main__":
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)
    download_all()