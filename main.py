import torch
import numpy as np
from icecream import ic
import rlcard
from rlcard.agents import RandomAgent
from rlcard import *
from rlcard.envs.uno import UnoEnv
DEFAULT_GAME_CONFIG = {
    'game_num_players': 2,
}

env = UnoEnv(False)
env.set_agents([RandomAgent(num_actions=61)])

print(env.num_actions)  # 2
print(env.num_players)  # 1
print(env.state_shape)  # [[2]]
print(env.action_shape)  # [None]

trajectories, payoffs = env.run()
ic(trajectories)
