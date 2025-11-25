# dataset.py
import torch
import glob
import os
import numpy as np
from torch.utils.data import Dataset

class UnifiedMertDataset(Dataset):
    def __init__(self, feature_folder, labels_map, task_type, num_classes):
        self.files = glob.glob(f"{feature_folder}/*.pt")
        self.labels_map = labels_map
        self.task_type = task_type
        self.num_classes = num_classes
        
        # Filter files that actually have labels
        self.files = [f for f in self.files if self._get_key(f) in self.labels_map]

    def _get_key(self, filepath):
        # Returns "song.wav" from "/path/to/song.pt"
        return os.path.basename(filepath).replace('.pt', '.wav')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        filename = self._get_key(path)
        
        # 1. Load & Pool Features
        embedding = torch.load(path)
        if embedding.dim() == 3: embedding = embedding.squeeze(0)
        embedding_mean = embedding.mean(dim=0) # (768,)

        # 2. Get Label
        raw_label = self.labels_map[filename]

        # 3. Format Label based on Task Type
        if self.task_type == "multiclass":
            # Expects single integer: 5
            label_tensor = torch.tensor(raw_label, dtype=torch.long)
            
        elif self.task_type == "multilabel":
            # Expects list of indices: [0, 5] -> One-hot vector
            label_tensor = torch.zeros(self.num_classes, dtype=torch.float32)
            for cls_idx in raw_label:
                label_tensor[cls_idx] = 1.0
                
        elif self.task_type == "regression":
            # Expects float list: [0.5, 0.8]
            label_tensor = torch.tensor(raw_label, dtype=torch.float32)

        return embedding_mean, label_tensor