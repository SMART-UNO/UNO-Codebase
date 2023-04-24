from uno.envs.uno2penv import UnoEnv2P
import torch
import numpy as np
from uno.envs.unoenv import UnoEnv
from uno.agents.random_agent import RandomAgent
from torch import nn
from collections import deque
import random



class DQNAgent:
    def __init__(self, num_actions, state_size=240, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), gamma=0.99, learning_rate=0.001, batch_size=64, memory_size=10000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, update_frequency=1000):
        self.state_size = state_size
        self.use_raw = False
        self.num_actions = num_actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_size)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_frequency = update_frequency
        self.device = device

        self.q_network = QNetwork(num_actions=num_actions, state_size=state_size, device=device).to(device)
        self.target_network = QNetwork(num_actions=num_actions, state_size=state_size, device=device).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def step(self, state):
        # Reshape the state to the expected input size
        reshaped_state = state['obs'].reshape(-1)

        # if reshaped_state.size != self.state_size:
        #     reshaped_state = reshaped_state[:self.state_size] if reshaped_state.size > self.state_size else np.pad(
        #         reshaped_state, (0, self.state_size - reshaped_state.size))

        if np.random.rand() > self.epsilon:
            reshaped_state = torch.from_numpy(reshaped_state).float().unsqueeze(0).to(self.device)
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(reshaped_state)
            self.q_network.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.randint(self.num_actions)


    def store(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        # sample phase
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        # learn phase
        q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        q_expected = self.q_network(states).gather(1, actions)

        loss = self.loss_fn(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


class QNetwork(nn.Module):
    def __init__(self, num_actions=61, state_size=240, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), layers=2):
        super(QNetwork, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size
        self.layers = layers
        self.device = device

        layer_dims = [self.state_size] + [128] * self.layers
        fc = []
        for i in range(len(layer_dims) - 1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(layer_dims[-1], self.num_actions, bias=True))
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, s):
        return self.fc_layers(s)


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


