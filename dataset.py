# dataset.py
import os
import glob
import torch
from torch.utils.data import Dataset

class MertEmbeddingsDataset(Dataset):
    def __init__(self, feature_folder, labels_dict):
        self.files = glob.glob(f"{feature_folder}/*.pt")
        self.labels = labels_dict # Dict mapping filename -> integer label
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        filename = os.path.basename(path).replace('.pt', '.wav')
        
        # Load Tensor
        embedding = torch.load(path) # Shape: (1, 375, Hidden_Dim)
        embedding = embedding.squeeze(0) # Shape: (375, Hidden_Dim)
        
        # Get Label (default to 0 if not found)
        label = self.labels.get(filename, 0) 
        
        return embedding, label