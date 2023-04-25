import numpy as np
import torch
import torch.nn as nn
from icecream import ic
# External import
from model.mc_backbone import MC_Q
from utils import DEVICE
torch.manual_seed(2023)
np.random.seed(2023)


class MCAgent(object):

    def __init__(self, num_actions, lr=1e-4, eps=0.05, df=0.95):
        self.name = "MC Agent"
        self.use_raw = False
        self.num_actions = num_actions
        # Q-Value estimation network
        self.Q = MC_Q(512, 61).to(DEVICE)
        # Optimizer
        self.opt = torch.optim.Adam(self.Q.parameters(), lr=lr)
        # Hyperparameters
        self.eps = eps
        self.df = df

    @staticmethod
    def random_action(legal_actions):
        # This is used for eps-greedy exploration
        return np.random.choice(legal_actions)

    def step(self, state):
        # Obtain legal actions
        # ic(state['obs'].shape)
        legal_actions = list(state['legal_actions'].keys())
        # Obtain action values by approximation
        val_lst = self.Q(state['obs'])[legal_actions]
        assert len(legal_actions) == len(val_lst)
        # Action
        rand_val = np.random.rand()
        assert rand_val >= 0 and rand_val <= 1
        if rand_val < self.eps:
            return self.random_action(legal_actions)
        else:
            # ic(torch.argmax(val_lst).item())
            return legal_actions[torch.argmax(val_lst).item()]

    def eval_step(self, state, is_greedy=False):
        # Return optimal policy based on LEARNED policy
        legal_actions = list(state['legal_actions'].keys())
        # Obtain action values by approximation
        val_lst = self.Q(state['obs'])[legal_actions]
        assert len(legal_actions) == len(val_lst)
        # Action
        rand_val = np.random.rand()
        assert rand_val >= 0 and rand_val <= 1
        # eps = self.eps if not is_greedy else 1
        if rand_val < self.eps:
            return self.random_action(legal_actions), None
        else:
            # ic(torch.argmax(val_lst).item())
            return legal_actions[torch.argmax(val_lst).item()], None

    def train(self, n=1000):
        pass
        # Loop through episodes
        # Generate episodes
        # Loop through each step
        # Compute Gt
        # self.optimizer.zero_grad()
        # loss = (Gt - self.Q[state][action])**2
        # in theory: w = w - lr * (-2 * (Gt - self.Q[state][action]) * dQ/dw)
        # loss.backward()
        # self.optimizer.step()
