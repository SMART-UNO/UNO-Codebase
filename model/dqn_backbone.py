import torch
from torch import nn

from collections import deque
import random
class QNetwork(nn.Module):
    def __init__(self, num_actions=61, state_size=240, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), layers=2):
        super(QNetwork, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size
        self.layers = layers
        self.device = device
        hidden_dim1 = 128
        hidden_dim2 = 512
        # Can add one more layer for testing purposes
        self.fc_layers = nn.Sequential(
            nn.Linear(state_size, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, num_actions)
        )

    def forward(self, state):
        state = state.view(state.size(0), -1)  # Flatten the input tensor
        return self.fc_layers(state)


class ReplayMemory:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def store(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


