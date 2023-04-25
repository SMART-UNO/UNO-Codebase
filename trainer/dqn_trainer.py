import numpy as np
import torch
from tqdm import tqdm
from uno.envs.uno2penv import UnoEnv2P
from uno.agents.random_agent import RandomAgent
from uno.agents.dqn_agent import DQNAgent
from tests.eval import test_trained_agents, plot_avg_rewards

# Hyperparameter declaration
num_episodes = 1000
learning_rate = 1e-4
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
gamma = 0.95
update_frequency = 1000

# Initialize agent
base_agent = RandomAgent(61)
dqn_agent = DQNAgent(num_actions=61, learning_rate=learning_rate, epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_decay=epsilon_decay, gamma=gamma)

# Initialize environment
env = UnoEnv2P(base_agent, dqn_agent)

# Train
for episode in tqdm(range(num_episodes)):
    env.reset()
    state = env.cur_state
    done = False

    while not done:
        action = dqn_agent.step(state)
        next_state, is_over = env.step(action)
        reward = env.get_payoffs()[1]

        dqn_agent.store(state, action, reward, next_state, is_over)
        dqn_agent.update()

        state = next_state
        done = is_over

        if episode % update_frequency == 0:
            dqn_agent.update_target_network()


# Evaluation
# n = 1000
# env = UnoEnv2P(RandomAgent(61), dqn_agent)
# payoffs_lst, trajectories_lst = [], []
#
# for idx in tqdm(range(n)):
#     env.reset()
#     trajectories, payoffs = env.run()
#     payoffs_lst.append(payoffs)
#     trajectories_lst.append(trajectories)


test_trained_agents(dqn_agent, base_agent, 10000, True)
# test_trained_agents(base_agent, dqn_agent, 10000, True)
