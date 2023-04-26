import numpy as np
import torch
import torch.nn as nn
from icecream import ic
from tqdm import tqdm
# External import
from model.mc_backbone import MC_Q
from uno.envs.unoenv import UnoEnv
from utils import DEVICE
torch.manual_seed(2023)
np.random.seed(2023)


class MCAgent(object):

    def __init__(self, num_actions, env, lr=1e-4, eps=0.05, df=0.95):
        self.env = env
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
            return self.random_action(legal_actions)
        else:
            # ic(torch.argmax(val_lst).item())
            return legal_actions[torch.argmax(val_lst).item()]

    def train(self, n=1000):
        '''
        Function to train a Monte Carlo Agent.

        The algorithm is as follows:
            Loop through episodes
                Generate episodes
                Loop through each step
                    Compute Gt
                    self.optimizer.zero_grad()
                    loss = (Gt - self.Q[state][action])**2
                        in theory: w = w - lr * (-2 * (Gt - self.Q[state][action]) * dQ/dw)
                    loss.backward()
                    self.optimizer.step()
        '''
        for _ in tqdm(range(n)):
            states, actions, payoff = self.env.run_monte_carlo()

            G_t = 0
            T = len(actions)  # total number of steps taken
            for t in range(T):
                # only training the first player
                G_t += pow(self.df, T - t - 1) * payoff[0]
                self.opt.zero_grad()
                # ic(states[0].keys())
                # ic(actions[0])
                # Probably should be the following?
                loss = (G_t - self.Q(states[t]['obs'])[actions[t]])**2
                # loss = (G_t - self.Q[states[0][t]][actions[0][t]])**2
                loss.backward()
                self.opt.step()


# monte_carlo = MCAgent(61)
# state = torch.zeros((4, 4, 15))
# out = monte_carlo.Q(state)
