# model.py
import torch
import torch.nn as nn
import config

class DownstreamHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1) 
        self.mlp = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # x shape: (Batch, Time, Dim) -> (Batch, Dim, Time)
        x = x.transpose(1, 2) 
        x = self.pool(x).squeeze(2)
        return self.mlp(x)