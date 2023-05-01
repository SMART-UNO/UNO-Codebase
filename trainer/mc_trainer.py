import numpy as np
import torch
import torch.nn as nn
from icecream import ic
from tqdm import tqdm
from tests.eval import test_trained_agents, plot_avg_rewards
from utils import parse_payoffs, DEVICE
from uno.agents.mc_agent import MCAgent
from uno.agents.random_agent import RandomAgent
from model.mc_backbone import MC_Q
from uno.envs.unoenv import UnoEnv
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# External import

torch.manual_seed(4529)
np.random.seed(4529)

# -------------------- Hyperparameter Declaration -------------------- #
n = 200000
lr = 1e-4
eps = 0.95
update_eps_every_n = 2000
decay_rate = 0.96
discount_factor = 0.95

# --------------------- Environment Declaration --------------------- #
env = UnoEnv(False)
avg_payoff_mc_first, avg_payoff_mc_second = [], []

# ------------------------ Agent Declaration ------------------------ #
mc_agent = MCAgent(num_actions=61, env=env,
                   lr=lr, eps=eps, df=discount_factor)
# mc_agent_second = MCAgent(num_actions=61, env=env,
#                           order=1, lr=lr, eps=eps, df=discount_factor)
random_agent = RandomAgent(61)
env.set_agents([mc_agent, random_agent])

# --------------------- Statistics --------------------- #
eval_every_n = 1000
# avg_payoff_mc_first, avg_payoff_mc_second = [], []

# --------------------- Training Code --------------------- #

for episode in tqdm(range(n)):
    mc_agent.train()
    if (episode + 1) % update_eps_every_n == 0:
        mc_agent.update_eps(decay_rate)

    if (episode + 1) % eval_every_n == 0:
        r_mc_first, _ = test_trained_agents(
            mc_agent, random_agent, 1000, False)
        _, r_mc_second = test_trained_agents(
            random_agent, mc_agent, 1000, False)
        avg_payoff_mc_first.append((episode, r_mc_first))
        avg_payoff_mc_second.append((episode, r_mc_second))

# --------------------- Final Evaluation --------------------- #
test_trained_agents(mc_agent, random_agent, 10000, True)
test_trained_agents(random_agent, mc_agent, 10000, True)

# print(reward_mc_first, reward_mc_second)


# --------------------- Plot Results ---------------------
# --------------------- Plot Results --------------------- #
plt_path = f"log/MC/mc-agent-[{n}]-[{lr}]-[{eps}]-[{discount_factor}]"
plot_avg_rewards(avg_payoff_mc_first,
                 "Average Rewards (MC Agent Plays First)",
                 plt_path + "-[first].png")
plot_avg_rewards(avg_payoff_mc_second,
                 "Average Rewards (MC Agent Plays Second)",
                 plt_path + "-[second].png")
