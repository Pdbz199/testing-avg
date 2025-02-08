import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()

        self.q_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, x):
        return self.q_net(x)
