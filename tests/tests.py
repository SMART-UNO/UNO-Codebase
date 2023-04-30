import numpy as np
import torch
import torch.nn as nn
from icecream import ic
import pandas as pd
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

# TODO: choose best agents and add to here...
CHECKPOINTS = {
    "Random Agent": RandomAgent(61),
    "SARSA Agent": torch.load(
        "checkpoint/SARSA/sarsa-agent-[50000]-[0.0001]-[0.05]-[0.95].pt", map_location=DEVICE),
    "REINFORCE Agent": torch.load(
        "checkpoint/REINFORCE/reinforce-agent-[50000]-[0.0001]-[0.95]-stanford.pt"),
    "DQN Agent": torch.load(
        "checkpoint/SARSA/sarsa-agent-[50000]-[0.0001]-[0.05]-[0.95].pt", map_location=DEVICE),
    "MC Agent": torch.load(
        "checkpoint/SARSA/sarsa-agent-[50000]-[0.0001]-[0.05]-[0.95].pt", map_location=DEVICE)
}


def contests(n=10000, seed=2023):
    np.random.seed(seed)
    stats = pd.DataFrame(columns=CANDIDATES, index=CANDIDATES)
    for agent1 in CANDIDATES:
        for agent2 in CANDIDATES:
            agent1_ckpt = CHECKPOINTS[agent1]
            # ic(agent1_ckpt)
            agent2_ckpt = CHECKPOINTS[agent2]
            # add necessary parameters to avoid bugs
            agent1_ckpt.eps, agent2_ckpt.eps = 0.01, 0.01
            # For testing purpose only (remove this line later)
            setattr(agent1_ckpt, "name", agent1)
            setattr(agent2_ckpt, "name", agent2)
            # Test
            r, _ = test_trained_agents(
                agent1_ckpt, agent2_ckpt, n, False)
            # ic(r)
            agent1_win_rate = 0.5 + r / 2
            # ic(agent1_win_rate)
            stats.loc[agent1, agent2] = f"{(agent1_win_rate * 100):.2f}%"
    return stats


if __name__ == "__main__":
    # np.random.seed(2023)
    # np.random.seed(4529)
    # Test sarsa agent
    # random_agent = RandomAgent(61)
    # sarsa_agent = torch.load("checkpoint/SARSA/sarsa-agent-[100000]-[0.0001]-[0.05]-[1].pt",
    #  map_location=torch.device('cpu'))

    # reinforce_agent = torch.load(
    #     "checkpoint/REINFORCE/reinforce-agent-[50000]-[0.0001]-[0.95]-stanford.pt")
    # sarsa_agent = torch.load(
    #     "checkpoint/SARSA/sarsa-agent-[50000]-[0.0001]-[0.05]-[0.95].pt", map_location=DEVICE)

    # sarsa_agent.eps = 0.01
    # reinforce_agent.eps = 0.01
    # # For testing purpose only (remove this line later)
    # setattr(sarsa_agent, "name", "SARSA Agent")
    # setattr(reinforce_agent, "name", "REINFORCE Agent")
    # # Test reinforce vs. sarsa
    # test_trained_agents(reinforce_agent, sarsa_agent, 10000, True)
    # test_trained_agents(sarsa_agent, reinforce_agent, 10000, True)

    # Test sarsa against random
    # test_trained_agents(random_agent, sarsa_agent, 10000, True)
    # test_trained_agents(sarsa_agent, random_agent, 10000, True)

    # TEST
    stats = contests(n=1000)
    print(stats)
