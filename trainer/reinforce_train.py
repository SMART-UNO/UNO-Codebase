import matplotlib.pyplot as plt
from utils import parse_payoffs
from uno.envs.uno import UnoEnv
from uno.agents.random_agent import RandomAgent
from uno.agents.reinforce_agent import ReinforceAgent
from uno.game.uno.utils import ACTION_SPACE, ACTION_LIST
import sys
import torch
import numpy as np
from icecream import ic
from tqdm import tqdm
import pickle
import json

lr = 0.005
gamma = 0.9999

sys.path.insert(0, 'UNO-Codabase')

# All of imports here are CUSTOMIZED. We are not using RLCards anymore.

np.random.seed(2023)

num_obs = 4 * 4 * 15
num_actions = 61


def train_reinforce(n=100, lr=0.05, gamma=0.99):
    env = UnoEnv(False)

    reinforce_agent = ReinforceAgent(num_obs=num_obs, num_actions=num_actions)
    nn = reinforce_agent.nn

    env.set_agents([reinforce_agent, RandomAgent(num_actions=num_actions)])
    payoffs_lst, trajectories_lst = [], []

    optim = torch.optim.Adam(nn.parameters(), lr=lr)

    for idx in tqdm(range(n)):
        trajectories, payoffs = env.run(is_training=True)

        payoffs_lst.append(payoffs)
        trajectories_lst.append(trajectories)

        reward = payoffs[0]

        run = trajectories[0]
        t = len(run)//2
        States = run[:-1:2]
        Actions = run[1::2]

        DiscountedReturns = [gamma**k * reward for k in range(t-1, -1, -1)]

        for state, action, G in zip(States, Actions, DiscountedReturns):
            obs = torch.flatten(torch.tensor(state['obs'])).float()
            probs = nn(obs)
            dist = torch.distributions.Categorical(probs=probs)
            log_prob = dist.log_prob(torch.tensor(action))

            loss = - log_prob*G

            optim.zero_grad()
            loss.backward()
            optim.step()
    return payoffs_lst


payoffs = np.array(train_reinforce(n=1000))
cum_payoffs = np.cumsum(np.array(payoffs), axis=0)

plt.plot(cum_payoffs[:, 0])
plt.title('cumulative payoffs')
plt.xlabel("n° run")
plt.show()
