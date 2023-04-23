import numpy as np
import torch

class ReinforceAgent(object):

    def __init__(self, num_obs, num_actions):
        ''' Initilize the random agent
        Args:
            num_actions (int): The size of the ouput action space
        '''
        self.use_raw = False
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.nn = torch.nn.Sequential(
                    torch.nn.Linear(self.num_obs, 120),
                    torch.nn.ReLU(),
                    torch.nn.Linear(120, num_actions),
                    torch.nn.Softmax(dim=-1)
                )

    def step(self, state):
        ''' Predict the action given the curent state in gerenerating training data.
        Args:
            state (dict): An dictionary that represents the current state
        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        '''
        obs = torch.flatten(torch.tensor(state['obs'])).float()
        probs = self.nn(obs)
        legal_actions = list(state['legal_actions'])
        choice = legal_actions[torch.argmax(probs[legal_actions])]
        return choice

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function
        Args:
            state (dict): An dictionary that represents the current state
        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        obs = torch.flatten(torch.tensor(state['obs'])).float()
        probs = self.nn(obs)
        legal_actions = list(state['legal_actions'])
        choice = legal_actions[torch.argmax(probs[legal_actions])]

        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: probs[legal_actions[i]] for i in range(len(legal_actions))}

        return choice, info
