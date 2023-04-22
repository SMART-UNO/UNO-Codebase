# AUTHOR: Xiaoyang Song
import numpy as np
import torch
import torch.nn as nn
from icecream import ic
# External import
from uno.envs.uno2penv import UnoEnv2P
from model.sarsa_backbone import SARSA_Q
from uno.agents.random_agent import RandomAgent
from uno.agents.sarsa_agent import SARSAAgent

# Agent declaration
base_agent = RandomAgent(61)
sarsa_agent = SARSAAgent(61)
# Environment declaration
env = UnoEnv2P(base_agent, sarsa_agent)
# Hyperparameter declaration
num_episodes = 1
T = 100

for episode in range(num_episodes):
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
        # s = s_
        # if done:
    # if episode % log_internval == 0:  # test
    #     total_reward = 0.0
    #     for i in range(10):
    #         t_s = env.reset()
    #         t_r = 0.0
    #         tr = 0.0
    #         time = 0
    #         while(time < 300):
    #             time += 1
    #             qs = sarsa.net(th.Tensor(t_s))
    #             a = sarsa.greedy_action(qs)
    #             ts_, tr, tdone, _ = env.step(a.tolist())
    #             t_r += tr
    #             if tdone:
    #                 break
    #             t_s = ts_
    #         total_reward += t_r
    #     print("episode:"+format(episode)+",test score:"+format(total_reward/10))
