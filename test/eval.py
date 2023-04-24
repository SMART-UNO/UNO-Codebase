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
from utils import parse_payoffs, get_average_payoffs, DEVICE, line


def test_trained_agents(agent1, agent2, n):
    env = UnoEnv(False)
    # Change this later
    # agent2.Q.eval()
    env.set_agents([agent1, agent2])
    # Store statistics
    payoffs_lst, trajectories_lst = [], []

    for idx in tqdm(range(n)):
        env.reset()
        trajectories, payoffs = env.run()
        payoffs_lst.append(payoffs)
        trajectories_lst.append(trajectories)
    # Compute average rewards
    get_average_payoffs(payoffs_lst, True)
    # Print out statistics
    parse_payoffs(payoffs_lst, True)
