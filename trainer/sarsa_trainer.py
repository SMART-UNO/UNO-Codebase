# AUTHOR: Xiaoyang Song
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
from utils import parse_payoffs


# Agent declaration
base_agent = RandomAgent(61)
sarsa_agent = SARSAAgent(61)
# Environment declaration
env = UnoEnv2P(base_agent, sarsa_agent)
# Hyperparameter declaration
num_episodes = 1000
T = 1000

for episode in tqdm(range(num_episodes)):
    env.reset()
    t = 0
    r = 0.0
    # Initialize S & A
    S = env.cur_state
    A = sarsa_agent.step(S)
    while t < T:
        t += 1
        S_NEW, is_over = env.step(A)
        R = env.get_payoffs()[1]
        A_NEW = sarsa_agent.update(S, A, S_NEW, R, is_over)
        # Update
        A = A_NEW
        S = S_NEW
        if is_over:
            break


n = 100
env = UnoEnv(False)
env.set_agents([RandomAgent(num_actions=61), sarsa_agent])
# Store statistics
payoffs_lst, trajectories_lst = [], []

for idx in tqdm(range(n)):
    env.reset()
    trajectories, payoffs = env.run()
    payoffs_lst.append(payoffs)
    trajectories_lst.append(trajectories)
# Print out statistics
parse_payoffs(payoffs_lst, True)
