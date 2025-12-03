import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import UnifiedMertDataset
from model import DownstreamHead
import config
import pandas as pd
import os
import glob
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np

# --- MUSIC THEORY UTILS (For GiantSteps) ---
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def get_key_class_id(key_str):
    """Converts key string (e.g. 'C major') to class ID (0-23)."""
    # Normalize
    key_str = key_str.strip().lower().replace('\t', ' ')
    
    # Handle flat/sharp equivalents
    replacements = {'db': 'c#', 'eb': 'd#', 'gb': 'f#', 'ab': 'g#', 'bb': 'a#'}
    for flat, sharp in replacements.items():
        if flat in key_str:
            key_str = key_str.replace(flat, sharp)

    parts = key_str.split()
    if len(parts) < 2: return None
    
    root = parts[0].upper()
    mode = parts[1] # 'major' or 'minor'
    
    if root not in PITCH_CLASSES: return None
    root_idx = PITCH_CLASSES.index(root)
    
    # Map: 0-11 Major, 12-23 Minor
    if 'maj' in mode: return root_idx
    elif 'min' in mode: return root_idx + 12
    return None

# --- LABEL LOADER ---
def load_labels(task_name):
    """Loads ground truth labels for the specific task."""
    conf = config.TASKS[task_name]
    labels = {}
    print(f"--- Loading Labels for {task_name} ---")
    
    # CASE 1: GIANTSTEPS (Key Detection)
    if task_name == "giantsteps":
        # Look for .key files recursively in raw_data
        # Structure is typically raw_data/giantsteps/audio/*.mp3 and annotations/keys/*.key
        raw_root = "./train_data/giantsteps" 
        key_files = glob.glob(os.path.join(raw_root, "**", "*.key"), recursive=True)
        
        print(f"Found {len(key_files)} annotation files (.key). Parsing...")
        
        for kf in key_files:
            try:
                with open(kf, 'r') as f:
                    key_text = f.read().strip()
                
                class_id = get_key_class_id(key_text)
                
                if class_id is not None:
                    # Map filename: '12345.LOFI.key' -> '12345.LOFI.wav'
                    wav_name = os.path.basename(kf).replace('.key', '.wav')
                    # Also handle mp3 name just in case extractor saved that way
                    # But dataset.py usually expects .wav keys
                    labels[wav_name] = class_id
            except Exception:
                continue

    # CASE 2: GTZAN (Genre)
    elif task_name == "gtzan":
        # Labels are in the folder structure (genres/rock/rock.001.wav)
        # We need to reconstruct this since we flattened features
        # Assuming extraction kept filenames like 'rock.00001.pt'
        # We define fixed mapping to ensure consistency
        genres = ["blues", "classical", "country", "disco", "hiphop", 
                  "jazz", "metal", "pop", "reggae", "rock"]
        genre_map = {g: i for i, g in enumerate(genres)}
        
        # We don't scan raw files here, we rely on feature filenames
        # But to populate 'labels', we need to know valid files. 
        # Let's scan features directly for GTZAN since filenames contain labels.
        feature_files = glob.glob(f"{conf['folder']}/*.pt")
        for f in feature_files:
            fname = os.path.basename(f)
            wav_name = fname.replace('.pt', '.wav')
            # Parse "rock.0001.pt" -> "rock"
            genre_str = fname.split('.')[0]
            if genre_str in genre_map:
                labels[wav_name] = genre_map[genre_str]
    
    # CASE 4: Custom Top-50 CSV (MTT)
    elif task_name == "mtt":
        if not os.path.exists(conf['csv']):
            print(f"Error: Run prepare_mtt.py first to generate {conf['csv']}")
            return {}
            
        df = pd.read_csv(conf['csv'])
        print(f"Loading MTT Top-50 labels from {conf['csv']}...")
        
        for _, row in df.iterrows():
            fname = row['filename']
            # Convert "0;5;10" -> [0, 5, 10]
            indices = [int(x) for x in str(row['tag_indices']).split(';')]
            labels[fname] = indices
    
# CASE 3: CSV Based (MTT, EmoMusic, VocalSet, etc.)
    elif "csv" in conf:
        if not os.path.exists(conf['csv']):
            print(f"Warning: CSV file {conf['csv']} not found.")
            return {}
            
        # Detect separator (Tab for TSV, Comma for CSV)
        sep = '\t' if conf['csv'].endswith('.tsv') else ','
        df = pd.read_csv(conf['csv'], sep=sep)
        
        # Normalize column names to lowercase
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Identify Filename Column (Scan for common names)
        path_col = next((c for c in ['filename', 'file', 'path', 'mp3_path', 'id'] if c in df.columns), None)
        
        if not path_col:
            print(f"Error: Could not find filename column in {conf['csv']}. Available: {list(df.columns)}")
            return {}

        print(f"Parsing labels from {conf['csv']}...")
        
        for _, row in df.iterrows():
            # Clean filename: Ensure it matches the .wav names in your feature folder
            raw_name = str(row[path_col])
            fname = os.path.basename(raw_name)
            if not fname.endswith('.wav'):
                fname = os.path.splitext(fname)[0] + '.wav'

            # Sub-case: Regression (EmoMusic)
            if conf['type'] == 'regression':
                arousal = row.get('arousal') or row.get('arousal_mean')
                valence = row.get('valence') or row.get('valence_mean')
                if arousal is not None and valence is not None:
                    labels[fname] = [float(arousal), float(valence)]

            # Sub-case: Classification (VocalSet, etc.)
            # Check for explicit 'label' column (integers)
            # Sub-case: Classification (Generic)
            # Use 'label_col' from config if it exists, otherwise default to 'label'
            target_col = conf.get('label_col', 'label')
            
            if target_col in df.columns:
                try:
                    labels[fname] = int(row[target_col])
                except ValueError:
                    continue
    
    print(f"Successfully loaded {len(labels)} labels.")
    return labels

def validate_multilabel(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            # Save for metrics
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate Metrics (Macro Average)
    try:
        roc_auc = roc_auc_score(all_targets, all_preds, average='macro')
        ap = average_precision_score(all_targets, all_preds, average='macro')
    except ValueError:
        roc_auc = 0
        ap = 0
        
    return total_loss / len(val_loader), roc_auc, ap

def train_task(task_name):
    if task_name not in config.TASKS:
        print(f"Error: Task {task_name} not found in config.py")
        return

    conf = config.TASKS[task_name]
    print(f"--- Training {task_name.upper()} ({conf['type']}) ---")

    # 1. Load Labels
    labels_map = load_labels(task_name)
    
    if len(labels_map) == 0:
        print(f"CRITICAL ERROR: No labels found for {task_name}.")
        print("Check if:")
        print(f"1. You have downloaded the raw data (including metadata/keys) to ./raw_data/{task_name}")
        print("2. You have run mert_extractor.py to generate feature files.")
        return

    # 2. Dataset
    try:
        dataset = UnifiedMertDataset(conf['folder'], labels_map, conf['type'], conf['classes'])
    except Exception as e:
        print(f"Dataset Init Error: {e}")
        return

    if len(dataset) == 0:
        print(f"Error: Dataset length is 0. Mismatch between Label filenames and Feature filenames?")
        print("Exiting.")
        return

    train_len = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_len, len(dataset)-train_len])
    
    # Drop last to avoid batch=1 errors
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, drop_last=False)

    # 3. Model & Loss
    model = DownstreamHead(conf['classes'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    if conf['type'] == "multiclass":
        criterion = nn.CrossEntropyLoss()
    elif conf['type'] == "multilabel":
        criterion = nn.BCEWithLogitsLoss()
    elif conf['type'] == "regression":
        criterion = nn.MSELoss()

# 4. Loop
    print(f"Training on {len(train_set)} samples, Validating on {len(val_set)} samples.")
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        train_loss = total_loss / len(train_loader)
        
        # Validation Step
        if conf['type'] == 'multilabel':
            val_loss, val_roc, val_ap = validate_multilabel(model, val_loader, criterion)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | ROC-AUC: {val_roc:.4f} | AP: {val_ap:.4f}")
        else:
            # Fallback for other tasks (add logic for multiclass acc later)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}")

    # 5. Save
    torch.save(model.state_dict(), f"./features/trained_{task_name}.pth")
    print(f"Saved trained_{task_name}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Name of task (giantsteps, gtzan, etc)")
    args = parser.parse_args()
    
    train_task(args.task)