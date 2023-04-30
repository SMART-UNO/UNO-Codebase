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
from tests.eval import test_trained_agents, plot_avg_rewards

# torch.manual_seed(4529)
# np.random.seed(4529)
torch.manual_seed(2023)
np.random.seed(2023)
# -------------------- Hyperparameter Declaration -------------------- #
num_episodes = 50000
lr = 1e-4

# Epsilons
eps = 0.05
update_eps_every_n = 1000
decay_rate = 1

# Discount Factor
discount_factor = 0.95
T = 10000  # Just some large number

# ------------------------ Agent Declaration ------------------------ #
base_agent = RandomAgent(61)
sarsa_agent = SARSAAgent(num_actions=61, lr=lr, eps=eps, df=discount_factor)

# --------------------- Environment Declaration --------------------- #
env = UnoEnv2P(base_agent, sarsa_agent)

# ----------------------- Checkpoint Loading ----------------------- #
# Load checkpoint if necessary
# checkpoint = "checkpoint/SARSA/sarsa-agent-[50000]-[0.0001]-[0.05]-[0.95].pt"
# checkpoint = "checkpoint/SARSA/sarsa-agent-[50000].pt"
checkpoint = None
if checkpoint is not None:
    sarsa_agent = torch.load(checkpoint,
                             map_location=DEVICE)
    ic('Checkpoint Loaded!')
    sarsa_agent.Q = sarsa_agent.Q.to(DEVICE)

# --------------------- Statistics --------------------- #
eval_every_n = 1000
avg_payoff_sarsa_first, avg_payoff_sarsa_second = [], []

# --------------------- Training Loop --------------------- #
for episode in tqdm(range(num_episodes)):
    sarsa_agent.Q.train()
    env.reset()
    t = 0
    # Initialize S & A
    S = env.cur_state
    A = sarsa_agent.step(S)
    # Handle base case where the game is over directly
    if env.is_over():
        continue

    # --------------------- Training within one episode --------------------- #
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

    # --------------------- Update Epsilon --------------------- #
    if (episode + 1) % update_eps_every_n == 0:
        sarsa_agent.update_eps(decay_rate)

    # --------------------- Evaluation every n episodes --------------------- #
    if (episode + 1) % eval_every_n == 0:
        r_sarsa_first, _ = test_trained_agents(
            sarsa_agent, base_agent, 1000, False)
        _, r_sarsa_second = test_trained_agents(
            base_agent, sarsa_agent, 1000, False)
        avg_payoff_sarsa_first.append((episode, r_sarsa_first))
        avg_payoff_sarsa_second.append((episode, r_sarsa_second))
        # Store back to agents
        sarsa_agent.eval_first.append((episode, r_sarsa_first))
        sarsa_agent.eval_first.append((episode, r_sarsa_second))

# --------------------- Save Training Checkpoint --------------------- #
torch.save(sarsa_agent,
           f"checkpoint/SARSA/sarsa-agent-[{num_episodes}]-[{lr}]-[{eps}]-[{discount_factor}].pt")

# --------------------- Final Evaluation --------------------- #
test_trained_agents(sarsa_agent, base_agent, 10000, True)
test_trained_agents(base_agent, sarsa_agent, 10000, True)

# --------------------- Plot Results --------------------- #
plt_path = f"log/SARSA/sarsa-agent-[{num_episodes}]-[{lr}]-[{eps}]-[{discount_factor}]"
plot_avg_rewards(avg_payoff_sarsa_first,
                 "Average Rewards (SARSA Agent Plays First)",
                 plt_path + "-[first].png")
plot_avg_rewards(avg_payoff_sarsa_second,
                 "Average Rewards (SARSA Agent Plays Second)",
                 plt_path + "-[second].png")
