import torch
import numpy as np
from icecream import ic
from tqdm import tqdm
import pickle
import json

# All of imports here are CUSTOMIZED. We are not using RLCards anymore.
from uno.envs.unoenv import UnoEnv
from uno.agents.random_agent import RandomAgent
from uno.agents.sarsa_agent import SARSAAgent
from utils import parse_payoffs

np.random.seed(2023)

# INITIALIZE environments (already overwritten)
env = UnoEnv(True)
env.set_agents([RandomAgent(num_actions=61),
               SARSAAgent(num_actions=61)])
# Test step by step
# state = env.step(qw)

# For testing, just two random agents
trajectories, payoffs = env.run()
ic(env.is_over())
ic(len(trajectories))
ic(payoffs)


######  TESTING  ######

# Check returned trajectories
# Just to get a sense of how "trajectories" are stored
check_trajectory = False
if check_trajectory:
    for idx, item in enumerate(trajectories[0]):
        ic(f"{idx} -- {type(item)}")
        if type(item) == np.int64:
            ic(item)
        else:
            ic(len(item['action_record']))


check_action_recorder = False
if check_action_recorder:
    # Check action_recorder functionality
    ic(trajectories[0][0]['action_record'])
    ic(trajectories[0][28]['action_record'])
    assert trajectories[0][28]['action_record'] == trajectories[0][0]['action_record']

check_transitions = False
if check_transitions:
    # Save trajectories to .json for testing purpose
    with open("log/test.json", "w") as f:
        trajectories[0][0]['obs'] = trajectories[0][0]['obs'].tolist()
        for item, val in trajectories[0][0].items():
            ic(f"{item} -- {type(val)}")
        json.dump(dict(trajectories[0][0]), f, separators=(',', ': '))


def test_random_player(n=100):
    env = UnoEnv(False)
    # For testing, just two random agents
    env.set_agents([RandomAgent(num_actions=61), RandomAgent(num_actions=61)])
    # Store statistics
    payoffs_lst, trajectories_lst = [], []

    for idx in tqdm(range(n)):
        env.reset()
        trajectories, payoffs = env.run()
        payoffs_lst.append(payoffs)
        trajectories_lst.append(trajectories)
    # Print out statistics
    parse_payoffs(payoffs_lst, True)


# Simulation results
# Total Number of Games: 10000
# P0 wins 5119 games (P0 win rate: 51.19%)
# P1 wins 4881 games (P1 win rate: 48.81%)
# Draws 0 games (Draw rate: 0.0%)

# test_random_player(10000)
