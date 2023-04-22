import numpy as np
import torch
import torch.nn as nn
from icecream import ic


# TODO:(Xiaoyang) REFACTOR THIS CLASS SOON...
# Note: if any of you want to modify this one, please lmk first.
class SARSAAgent(object):

    def __init__(self, num_actions):
        self.use_raw = False
        self.num_actions = num_actions
        # Q-Value estimation network: currently the architecture is borrowed from Stanford report
        self.Q = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 15, 512),
            nn.ReLU(),
            nn.Linear(512, 61)
        )

    def step(self, state):
        legal_actions = list(state['legal_actions'].keys())
        return np.random.choice(legal_actions)

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
    sarsa = SARSAAgent(61)
    state = torch.zeros((4, 4, 15))
    # Test nn
    out = sarsa.Q(state)
    ic(out.shape)
