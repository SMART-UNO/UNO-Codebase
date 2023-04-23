# AUTHOR: Xiaoyang Song
import numpy as np
import torch
import torch.nn as nn
from icecream import ic
# External import
from uno.envs.uno2penv import UnoEnv2P
from model.sarsa_backbone import SARSA_Q
torch.manual_seed(2023)
np.random.seed(2023)


class SARSAAgent(object):

    def __init__(self, num_actions, lr=1e-2, eps=0.05):
        self.use_raw = False
        self.num_actions = num_actions
        # Q-Value estimation network
        self.Q = SARSA_Q(512, 61)
        # Optimizer
        self.opt = torch.optim.Adam(self.Q.parameters(), lr=lr)
        # Scheduler (unnecessary for now)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, 5, gamma=0.1)
        self.criterion = nn.MSELoss()
        # Hyperparameters
        self.eps = eps
        self.df = 1

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

    def eval_step(self, state):
        # Return optimal policy based on LEARNED policy
        legal_actions = list(state['legal_actions'].keys())
        # Obtain action values by approximation
        val_lst = self.Q(state['obs'])[legal_actions]
        assert len(legal_actions) == len(val_lst)
        # Action
        rand_val = np.random.rand()
        assert rand_val >= 0 and rand_val <= 1
        if rand_val < self.eps:
            return self.random_action(legal_actions), None
        else:
            # ic(torch.argmax(val_lst).item())
            return legal_actions[torch.argmax(val_lst).item()], None

    def update(self, S, A, S_NEW, R, is_over):
        q_est = self.Q(S['obs'])[A]
        A_NEW = self.step(S_NEW)
        next_q = self.Q(S_NEW['obs'])[A_NEW]

        q_true = self.df * next_q + R if not is_over else R
        loss = (q_true - q_est)**2
        # ic(loss)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return A_NEW

        # Simple Test Code
if __name__ == '__main__':
    # Test sarsa agent
    sarsa = SARSAAgent(61)
    state = torch.zeros((4, 4, 15))
    # Test nn
    # ic(state.shape)
    out = sarsa.Q(state)
    # ic(out.shape)
    # ic(torch.sum(out))
    # Test UNO2PENV
    unoenv = UnoEnv2P(sarsa, sarsa)
    state = unoenv.get_state(1)
    ic(state)
    action = sarsa.step(state)
    ic(action)
    unoenv.step(action)
