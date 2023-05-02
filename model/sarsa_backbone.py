# AUTHOR: Xiaoyang Song
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from icecream import ic
from utils import DEVICE


class SARSA_Q(nn.Module):
    def __init__(self, hidden_dim=512, num_actions=61):
        super().__init__()
        # For now, assume simple FC network
        # Currently the architecture is borrowed from Stanford report
        self.model = nn.Sequential(
            nn.Linear(4 * 4 * 15, hidden_dim),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, state):
        # ic(state.shape)
        assert state.shape == torch.Size((4, 4, 15))
        if type(state) != torch.Tensor:
            state = torch.tensor(
                state, dtype=torch.float32, requires_grad=True)
        out = self.model(torch.flatten(state.to(DEVICE), 0, 2))
        return out
        # return torch.softmax(out, dim=-1) # We may not want to normalize here
