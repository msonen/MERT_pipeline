# dataset.py
import os
import glob
import torch
from torch.utils.data import Dataset

class MertEmbeddingsDataset(Dataset):
    def __init__(self, feature_folder, labels_dict):
        self.files = glob.glob(f"{feature_folder}/*.pt")
        self.labels = labels_dict 
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        filename = os.path.basename(path).replace('.pt', '.wav')
        
        # Load Tensor: Shape (1, Time_Steps, Dim) or (Time_Steps, Dim)
        embedding = torch.load(path)
        
        # Ensure it's squeezed to (Time, Dim) before averaging
        if embedding.dim() == 3:
            embedding = embedding.squeeze(0)
            
        # --- FIX: Average Pooling Here ---
        # We squash the Time dimension (dim=0) so every song becomes shape (Dim,)
        # e.g., (2244, 768) -> (768,)
        embedding_mean = embedding.mean(dim=0) 
        
        # Get Label
        label = self.labels.get(filename, 0) 
        
        return embedding_mean, label