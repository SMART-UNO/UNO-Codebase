import pickle
import torch
import numpy as np
from icecream import ic

# All of imports here are CUSTOMIZED. We are not using RLCards anymore.
from uno.envs.uno import UnoEnv
from uno.agents.random_agent import RandomAgent

np.random.seed(2023)
DEFAULT_GAME_CONFIG = {
    'game_num_players': 2,
}

env = UnoEnv(False)
# For testing, just two random agents
env.set_agents([RandomAgent(num_actions=61), RandomAgent(num_actions=61)])

print(env.num_actions)  # 2
print(env.num_players)  # 1
print(env.state_shape)  # [[2]]
print(env.action_shape)  # [None]

trajectories, payoffs = env.run()
ic(len(trajectories))
ic(payoffs)

# Check returned trajectories
# ic(trajectories[0])
ic(type(trajectories[0]))
ic(len(trajectories[0]))
ic(len(trajectories[1]))
ic(type(trajectories[0][0]))


# Just to get a sense of how "trajectories" are stored
for idx, item in enumerate(trajectories[1]):
    ic(f"{idx} -- {type(item)}")
    if type(item) == np.int64:
        ic(item)
    else:
        ic(len(item['action_record']))


# Save trajectories to .json for testing purpose
# with open("log/test.json", "wb") as f:
#     pickle.dump(trajectories[0], f)
