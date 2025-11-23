# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import config
from dataset import MertEmbeddingsDataset
from model import DownstreamHead

def train_downstream_task(feature_folder, labels_map, num_classes):
    # 1. Setup Data
    dataset = MertEmbeddingsDataset(feature_folder, labels_map)
    if len(dataset) == 0:
        print("No feature files found! Run mert_extractor.py first.")
        return
        
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # 2. Setup Model
    model = DownstreamHead(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training Loop
    print(f"--- Starting Training for {config.EPOCHS} Epochs ---")
    model.train()
    for epoch in range(config.EPOCHS):
        total_loss = 0
        for batch_embeds, batch_labels in loader:
            predictions = model(batch_embeds)
            loss = criterion(predictions, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{config.EPOCHS} | Loss: {total_loss:.4f}")
    
    return model

if __name__ == "__main__":
    # Example Labels (Filename -> Class ID)
    # In a real scenario, load this from a CSV file
    dummy_labels = {
        "sample_jazz.wav": 0,
        "sample_rock.wav": 1
    }
    
    train_downstream_task("./features", dummy_labels, num_classes=2)