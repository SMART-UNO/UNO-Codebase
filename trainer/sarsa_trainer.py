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
from utils import parse_payoffs, DEVICE
from tests.eval import test_trained_agents

torch.manual_seed(4529)
np.random.seed(4529)

# Hyperparameter declaration
num_episodes = 200000
lr = 1e-4
eps = 0.05
discount_factor = 0.95
T = 10000  # Just some large number

# Agent declaration
base_agent = RandomAgent(61)
sarsa_agent = SARSAAgent(num_actions=61, lr=lr, eps=eps, df=discount_factor)
# Environment declaration
env = UnoEnv2P(base_agent, sarsa_agent)

# Load checkpoint if necessary
# checkpoint = "checkpoint/SARSA/sarsa-agent-[50000]-[0.0001]-[0.05]-[0.95].pt"
# checkpoint = "checkpoint/SARSA/sarsa-agent-[50000].pt"
checkpoint = None
if checkpoint is not None:
    sarsa_agent = torch.load(checkpoint,
                             map_location=DEVICE)
    ic('Checkpoint Loaded!')
    sarsa_agent.Q = sarsa_agent.Q.to(DEVICE)

eval_every_n = 1000
# Training Loop
for episode in tqdm(range(num_episodes)):
    sarsa_agent.Q.train()
    env.reset()
    t = 0
    # Initialize S & A
    S = env.cur_state
    A = sarsa_agent.step(S)
    while t < T:
        t += 1
        S_NEW, is_over = env.step(A)
        R = env.get_payoffs()[1]
        # ic(R)
        A_NEW = sarsa_agent.update(S, A, S_NEW, R, is_over)
        # Update
        A = A_NEW
        S = S_NEW
        if is_over:
            break
    if (episode + 1) % eval_every_n == 0:
        test_trained_agents(sarsa_agent, base_agent, 100)
        test_trained_agents(base_agent, sarsa_agent, 100)

# Save results
torch.save(sarsa_agent,
           f"checkpoint/SARSA/sarsa-agent-[{num_episodes}]-[{lr}]-[{eps}]-[{discount_factor}].pt")

# Final Eval
test_trained_agents(sarsa_agent, base_agent, 10000)
test_trained_agents(base_agent, sarsa_agent, 10000)
