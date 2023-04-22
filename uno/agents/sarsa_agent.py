# AUTHOR: Xiaoyang Song
import numpy as np
import torch
import torch.nn as nn
from icecream import ic
# External import
from uno.envs.uno2penv import UnoEnv2P
from model.sarsa_backbone import SARSA_Q


class SARSAAgent(object):

    def __init__(self, num_actions, lr=1e-3, eps=0.05):
        self.use_raw = False
        self.num_actions = num_actions
        # Q-Value estimation network
        self.Q = SARSA_Q(512, 61)
        # Optimizer
        self.opt = torch.optim.Adam(self.Q.parameters(), lr=lr)
        # Scheduler (unnecessary for now)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, 5, gamma=0.1)
        # Hyperparameters
        self.eps = eps

    @staticmethod
    def random_action(legal_actions):
        # This is used for eps-greedy exploration
        return np.random.choice(legal_actions)

    def step(self, state):
        # Obtain legal actions
        ic(state['obs'].shape)
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
        probs = [0 for _ in range(self.num_actions)]
        for i in state['legal_actions']:
            probs[i] = 1/len(state['legal_actions'])

        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: probs[list(
            state['legal_actions'].keys())[i]] for i in range(len(state['legal_actions']))}

        return self.step(state), info


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
