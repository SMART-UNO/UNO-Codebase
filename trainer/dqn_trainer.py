import torch
from uno.agents.random_agent import RandomAgent
from uno.agents.dqn_agent import DQNAgent
from uno.envs.uno2penv import UnoEnv2P
from icecream import ic

num_actions = 61

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
base_agent = RandomAgent(num_actions)  # Initialize your base agent (RandomAgent) here
training_agent = DQNAgent(num_actions, device=device)  # Initialize your DQN training agent here
env = UnoEnv2P(base_agent, training_agent)

scores = training_agent.train(env)  # Pass the environment instance to the train method
