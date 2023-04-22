import numpy as np
import torch
import torch.nn as nn
from icecream import ic
# External import
from uno.envs.uno2penv import UnoEnv2P
from model.sarsa_backbone import SARSA_Q
from uno.agents.random_agent import RandomAgent
from uno.agents.sarsa_agent import SARSAAgent

# Test sarsa agent
sarsa = SARSAAgent(61)
rand = RandomAgent(61)
state = torch.zeros((4, 4, 15))
# Test nn
# ic(state.shape)
out = sarsa.Q(state)
# ic(out.shape)
# ic(torch.sum(out))
# Test UNO2PENV
unoenv = UnoEnv2P(rand, sarsa)
state = unoenv.get_state(1)
# ic(state)
action = sarsa.step(state)
ic(action)
unoenv.step(action)
