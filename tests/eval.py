import numpy as np
import torch
import torch.nn as nn
from icecream import ic
from tqdm import tqdm
from matplotlib import pyplot as plt
# External import
from uno.envs.unoenv import UnoEnv
from uno.envs.uno2penv import UnoEnv2P
from model.sarsa_backbone import SARSA_Q
from uno.agents.random_agent import RandomAgent
from uno.agents.sarsa_agent import SARSAAgent
from utils import parse_payoffs, get_average_payoffs, DEVICE, line


def test_trained_agents(agent1, agent2, n, verbose=False):
    env = UnoEnv(False)
    # Change this later
    names = [agent1.name, agent2.name]
    env.set_agents([agent1, agent2])
    # Store statistics
    payoffs_lst, trajectories_lst = [], []

    for idx in tqdm(range(n), disable=False):
        env.reset()
        trajectories, payoffs = env.run()
        payoffs_lst.append(payoffs)
        trajectories_lst.append(trajectories)
    # Compute average rewards
    r0, r1 = get_average_payoffs(payoffs_lst, names, verbose)
    # Print out statistics
    parse_payoffs(payoffs_lst, names, verbose)
    return r0, r1


def plot_avg_rewards(avg_payoff, title, filename):
    loi = np.array(avg_payoff)
    # ic(loi.shape)
    x, y = loi[:, 0], loi[:, 1]
    # plt.plot(x, y, color='#00b384')
    plt.plot(x, y, color='navy')
    plt.axhline(y=0.0, color='purple', linestyle='dashed')
    plt.xlabel("Iterations (Episodes)")
    plt.ylabel("Average Rewards")
    plt.title(title)
    plt.savefig(filename, dpi=400)
    plt.close()
