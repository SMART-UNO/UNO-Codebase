from uno.envs.uno2penv import UnoEnv2P
import torch
import numpy as np
from uno.envs.unoenv import UnoEnv
from uno.agents.random_agent import RandomAgent
from torch import nn

from model.dqn_backbone import QNetwork, ReplayMemory
from icecream import ic

class DQNAgent:
    def __init__(self, num_actions, state_size=240, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), gamma=0.99, learning_rate=0.001, batch_size=64, memory_size=10000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, update_frequency=1000):
        self.name = "DQN Agent"
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

        legal_actions = state['legal_actions']

        if np.random.rand() > self.epsilon:
            reshaped_state = torch.from_numpy(reshaped_state).float().unsqueeze(0).to(self.device)
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(reshaped_state)
            self.q_network.train()
            # Select the action with the highest Q-value from legal actions
            action = max(legal_actions.keys(), key=lambda x: action_values.cpu().data.numpy()[0][x])
        else:
            action = np.random.choice(list(legal_actions.keys()))

        return action

    def store(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        # sample phase
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states_obs = [state['obs'] for state in states]
        states = np.vstack(states_obs).reshape(self.batch_size, -1).astype(np.float32)
        states = torch.from_numpy(states).float().to(self.device)

        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)

        next_states_obs = [state['obs'] for state in next_states]
        next_states = np.vstack(next_states_obs).reshape(self.batch_size, -1).astype(np.float32)
        next_states = torch.from_numpy(next_states).float().to(self.device)

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

    def eval_step(self, state):
        state_obs = state['obs']
        state = torch.from_numpy(state_obs).float().unsqueeze(0).to(self.device)

        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train()

        return np.argmax(action_values.cpu().data.numpy()), None


