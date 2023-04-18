import pickle
import torch
import numpy as np
from icecream import ic
import json

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
# Just to get a sense of how "trajectories" are stored
for idx, item in enumerate(trajectories[0]):
    ic(f"{idx} -- {type(item)}")
    if type(item) == np.int64:
        ic(item)
    else:
        ic(len(item['action_record']))

# Check action recorder functionality
ic(trajectories[0][0]['action_record'])
ic(trajectories[0][28]['action_record'])
assert trajectories[0][28]['action_record'] == trajectories[0][0]['action_record']

# Save trajectories to .json for testing purpose
with open("log/test.json", "w") as f:
    trajectories[0][0]['obs'] = trajectories[0][0]['obs'].tolist()
    for item, val in trajectories[0][0].items():
        ic(f"{item} -- {type(val)}")
    json.dump(dict(trajectories[0][0]), f, separators=(',', ': '))
