# train_unified.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import UnifiedMertDataset
from model import DownstreamHead
import config
import pandas as pd
import os

def load_labels(task_name):
    """
    Helper to load CSV/Folder labels based on task config
    Returns: Dict { 'filename.wav': label_data }
    """
    conf = config.TASKS[task_name]
    labels = {}
    
    # CASE A: CSV Metadata (MTT, MTG, EmoMusic)
    if "csv" in conf:
        print(f"Loading labels from {conf['csv']}...")
        df = pd.read_csv(conf['csv'], sep='\t' if conf['csv'].endswith('tsv') else ',')
        
        # Example logic for MTT/MTG (Adjust column names to match your specific CSV!)
        for _, row in df.iterrows():
            fname = row['path'] # Column name depends on dataset
            if conf['type'] == 'multilabel':
                # Assuming tags are in a column 'tags' separated by comma
                # You need a mapping from "guitar" -> 5. Doing this dynamically:
                # (This is a placeholder, real implementation requires a tag vocabulary)
                pass 
            elif conf['type'] == 'regression':
                labels[fname] = [row['arousal'], row['valence']]
                
    # CASE B: Folder Structure (GTZAN, GiantSteps, VocalSet)
    else:
        # We assume simple subfolders or filename parsing
        # (Implemented generically here for brevity)
        pass
        
    return labels

def train_task(task_name):
    if task_name not in config.TASKS:
        print(f"Error: Task {task_name} not found in config.py")
        return

    conf = config.TASKS[task_name]
    print(f"--- Training {task_name.upper()} ({conf['type']}) ---")

    # 1. Load Labels (You need to implement the specific parsing logic for your CSVs)
    # For this example, let's assume labels_map is ready
    # labels_map = load_labels(task_name) 
    
    # DUMMY DATA for demonstration if you don't have the CSVs yet
    labels_map = {"test.wav": [0, 1]} if conf['type'] == 'multilabel' else {"test.wav": 0}

    # 2. Dataset
    dataset = UnifiedMertDataset(conf['folder'], labels_map, conf['type'], conf['classes'])
    train_len = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_len, len(dataset)-train_len])
    
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE)

    # 3. Model & Loss Selection
    model = DownstreamHead(conf['classes'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # CRITICAL: Select correct loss function
    if conf['type'] == "multiclass":
        criterion = nn.CrossEntropyLoss()
    elif conf['type'] == "multilabel":
        criterion = nn.BCEWithLogitsLoss() # For Multi-label (One-hot)
    elif conf['type'] == "regression":
        criterion = nn.MSELoss()           # For Regression

    # 4. Loop
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
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

    # 5. Save
    torch.save(model.state_dict(), f"trained_{task_name}.pth")
    print(f"Saved trained_{task_name}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Name of task from config.py (e.g., mtt, gtzan)")
    args = parser.parse_args()
    
    train_task(args.task)