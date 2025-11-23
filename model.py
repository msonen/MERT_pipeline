# model.py
import torch
import torch.nn as nn
import config

class DownstreamHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # REMOVED: self.pool = nn.AdaptiveAvgPool1d(1) 
        # The input is already pooled by dataset.py
        
        self.mlp = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Input x shape is now: (Batch, Dim)
        # No need to transpose or pool
        return self.mlp(x)