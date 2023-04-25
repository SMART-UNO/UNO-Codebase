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

# np.random.seed(2023)
# np.random.seed(4529)
# Test sarsa agent
random_agent = RandomAgent(61)
# sarsa_agent = torch.load("checkpoint/SARSA/sarsa-agent-[100000]-[0.0001]-[0.05]-[1].pt",
#  map_location=torch.device('cpu'))
sarsa_agent = torch.load(
    "checkpoint/SARSA/sarsa-agent-[50000]-[0.0001]-[0.05]-[0.95].pt", map_location=DEVICE)
sarsa_agent.eps = 0.01
# For testing purpose only (remove this line later)
setattr(sarsa_agent, "name", "SARSA Agent")

# Test
test_trained_agents(random_agent, sarsa_agent, 10000, True)
test_trained_agents(sarsa_agent, random_agent, 10000, True)
# test_trained_agents(random_agent, random_agent, 10000, True)

# Before Training
# Total Number of Games: 1000
# P0 wins 507 games (P0 win rate: 50.7%)
# P1 wins 493 games (P1 win rate: 49.3%)
# Draws 0 games (Draw rate: 0.0%)

# After 200000 episodes of training (still short training):
# Smartest agent: sarsa-agent-[50000]-[0.0001]-[0.05]-[0.95].pt
# SARSA plays second (SARSA is agent 0)
# SEED: 2023
# ------------------------------------------------------------
# Average Rewards
# ------------------------------------------------------------
# Agent 0 Average Reward: -0.1204
# Agent 1 Average Reward: 0.1204
# ------------------------------------------------------------
# Total Number of Games: 10000
# Agent 0 wins 4398 games (P0 win rate: 43.98%)
# Agent 1 wins 5602 games (P1 win rate: 56.02%)
# Draws 0 games (Draw rate: 0.0%)

# SEED: 4529
# ------------------------------------------------------------
# Average Rewards
# ------------------------------------------------------------
# Agent 0 Average Reward: -0.098
# Agent 1 Average Reward: 0.098
# ------------------------------------------------------------
# Total Number of Games: 10000
# Agent 0 wins 4510 games (P0 win rate: 45.1%)
# Agent 1 wins 5490 games (P1 win rate: 54.9%)
# Draws 0 games (Draw rate: 0.0%)

# SARSA plays first (SARSA is agent 0)
# SEED: 4529
# ------------------------------------------------------------
# Average Rewards
# ------------------------------------------------------------
# Agent 0 Average Reward: 0.1428
# Agent 1 Average Reward: -0.1428
# ------------------------------------------------------------
# Total Number of Games: 10000
# Agent 0 wins 5714 games (P0 win rate: 57.14%)
# Agent 1 wins 4286 games (P1 win rate: 42.86%)
# Draws 0 games (Draw rate: 0.0%)
