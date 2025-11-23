# train_gtzan.py
import glob
import os
from train import train_downstream_task  # Imports the function from your existing train.py
import torch

# --- CONFIGURATION ---
FEATURE_FOLDER = "./features/gtzan"  # Must match output of extractor

def prepare_gtzan_labels(folder_path):
    """
    Scans the folder of .pt files and generates a label map
    based on the GTZAN filename convention: 'genre.id.pt'
    """
    files = glob.glob(f"{folder_path}/*.pt")
    
    if not files:
        raise FileNotFoundError(f"No .pt files found in {folder_path}. Run mert_extractor.py first!")

    labels_map = {}
    genres_set = set()
    
    # 1. Parse Filenames
    for file_path in files:
        filename = os.path.basename(file_path) # e.g., "metal.00050.pt"
        
        # GTZAN convention: "genre.id.pt"
        parts = filename.split('.')
        genre = parts[0] # "metal"
        
        # Store for mapping
        wav_filename = filename.replace('.pt', '.wav') # Dataset loader expects .wav keys
        labels_map[wav_filename] = genre
        genres_set.add(genre)

    # 2. Create Integer IDs (0=blues, 1=classical, etc.)
    sorted_genres = sorted(list(genres_set))
    genre_to_id = {g: i for i, g in enumerate(sorted_genres)}
    
    print(f"Detected {len(sorted_genres)} Genres: {genre_to_id}")
    
    # 3. Convert map values to integers
    final_labels_map = {k: genre_to_id[v] for k, v in labels_map.items()}
    
    return final_labels_map, len(sorted_genres)

if __name__ == "__main__":
    print("--- 🎵 Setting up GTZAN Training 🎵 ---")
    
    # 1. Generate Label Map
    try:
        labels, num_classes = prepare_gtzan_labels(FEATURE_FOLDER)
        
        # 2. Call the generic training function
        # This uses the code you already have in train.py
        model = train_downstream_task(FEATURE_FOLDER, labels, num_classes)
        torch.save(model.state_dict(), "trained_genre_head.pth")
        print("Model saved to trained_genre_head.pth")


    except Exception as e:
        print(f"Setup Failed: {e}")