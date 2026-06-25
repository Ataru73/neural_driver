import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim=8, output_dim=9):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)
