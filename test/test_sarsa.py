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
from eval import *

np.random.seed(2023)
# Test sarsa agent
agent1 = RandomAgent(61)
# sarsa_agent = torch.load("checkpoint/SARSA/sarsa-agent-[100000]-[0.0001]-[0.05]-[1].pt",
#                          map_location=torch.device('cpu'))
sarsa_agent = torch.load(
    "checkpoint/SARSA/sarsa-agent-[50000].pt", map_location=DEVICE)
test_trained_agents(agent1, sarsa_agent, 1000)

# Before Training
# Total Number of Games: 1000
# P0 wins 507 games (P0 win rate: 50.7%)
# P1 wins 493 games (P1 win rate: 49.3%)
# Draws 0 games (Draw rate: 0.0%)

# SARSA test results: n = 50000 (short training)
# Total Number of Games: 1000
# P0 wins 452 games (P0 win rate: 45.2%)
# P1 wins 548 games (P1 win rate: 54.8%)
# Draws 0 games (Draw rate: 0.0%)
