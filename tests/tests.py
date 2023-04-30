import numpy as np
import torch
import torch.nn as nn
from icecream import ic
from tqdm import tqdm
# External import
from uno.envs.unoenv import UnoEnv
from uno.envs.uno2penv import UnoEnv2P
from model.sarsa_backbone import SARSA_Q
from uno.agents.random_agent import RandomAgent
from uno.agents.sarsa_agent import SARSAAgent
from uno.agents.reinforce_agent import ReinforceAgent
from eval import *


CANDIDATES = ["Random Agent", "SARSA Agent",
              "MC Agent", "REINFORCE Agent", "DQN Agent"]

CHECKPOINTS = {
    "Random Agent": RandomAgent(61),
    "SARSA Agent": torch.load(
        "checkpoint/SARSA/sarsa-agent-[50000]-[0.0001]-[0.05]-[0.95].pt", map_location=DEVICE),
    "REINFORCE Agent": torch.load(
        "checkpoint/REINFORCE/reinforce-agent-[50000]-[0.0001]-[0.95]-stanford.pt"),
    "DQN Agent": None,
    "MC Agent": None
}

np.random.seed(2023)
# np.random.seed(4529)
# Test sarsa agent
random_agent = RandomAgent(61)
# sarsa_agent = torch.load("checkpoint/SARSA/sarsa-agent-[100000]-[0.0001]-[0.05]-[1].pt",
#  map_location=torch.device('cpu'))

reinforce_agent = torch.load(
    "checkpoint/REINFORCE/reinforce-agent-[50000]-[0.0001]-[0.95]-stanford.pt")
sarsa_agent = torch.load(
    "checkpoint/SARSA/sarsa-agent-[50000]-[0.0001]-[0.05]-[0.95].pt", map_location=DEVICE)

sarsa_agent.eps = 0.01
reinforce_agent.eps = 0.01
# For testing purpose only (remove this line later)
setattr(sarsa_agent, "name", "SARSA Agent")
setattr(reinforce_agent, "name", "REINFORCE Agent")
# Test reinforce vs. sarsa
test_trained_agents(reinforce_agent, sarsa_agent, 10000, True)
test_trained_agents(sarsa_agent, reinforce_agent, 10000, True)

# Test sarsa against random
# test_trained_agents(random_agent, sarsa_agent, 10000, True)
# test_trained_agents(sarsa_agent, random_agent, 10000, True)
