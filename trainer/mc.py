import torch
import numpy as np
from icecream import ic
from tqdm import tqdm

from uno.envs.unoenv import UnoEnv
from scipy.stats import uniform

from torch import nn

env = UnoEnv(False)


class EstNetwork(nn.Module):
    '''
    Uses Pytorch for a neural network for q-value estimates
    '''

    def __init_(self, num_actions=61, state_size=241, layers=2):
        super(EstNetwork, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size
        self.layers = 2

        # build the Q network
        layer_dims = [self.state_shape] + self.layers
        fc = [nn.Flatten()]
        fc.append(nn.BatchNorm1d(layer_dims[0]))
        for i in range(len(layer_dims)-1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(layer_dims[-1], self.num_actions, bias=True))
        self.fc_layers = nn.Sequential(*fc)

    def next(self, s):
        '''
        Predicts the values 
        '''
        return self.fc_layers(s)


class DMCEstimator(object):

    def __init_(self, num_actions=61, state_size=241, layers=2, learning_rate=0.001):
        self.num_actions = num_actions
        self.state_size = state_size
        self.layers = layers
        self.learning_rate = learning_rate

        # building model
        q_model = EstNetwork(num_actions=num_actions,
                             state_size=state_size, layers=layers)
        q_model = q_model.to(self.device)
        self.q_model = q_model
        self.q_model.eval()

        # setting weights with xavier?
        for w in self.q_model.parameters():
            if len(w.data.shape) > 1:
                nn.init.xavier_uniform_(w.data)

        # setting up loss
        self.loss = nn.MSELoss(reduction='mean')

        # setting up learning process
        self.optimizer = torch.optim.Adam(
            self.q_model.parameters(), lr=self.learning_rate)

    def predict(self, s):
        ''' Predicts action values, but prediction is not included
              in the computation graph.  It is used to predict optimal next
              actions in the Double-DQN algorithm.
          Args:
            s (np.ndarray): (batch, state_len)
          Returns:
            np.ndarray of shape (batch_size, NUM_VALID_ACTIONS) containing the estimated
            action values.
        '''
        with torch.no_grad():
            s = torch.from_numpy(s).float().to(self.device)
            q_as = self.qnet(s).cpu().numpy()
        return q_as

    def update(self, s, a, y):
        ''' Updates the estimator towards the given targets.
            In this case y is the target-network estimated
            value of the Q-network optimal actions, which
            is labeled y in Algorithm 1 of Minh et al. (2015)
        Args:
          s (np.ndarray): (batch, state_shape) state representation
          a (np.ndarray): (batch,) integer sampled actions
          y (np.ndarray): (batch,) value of optimal actions according to Q-target
        Returns:
          The calculated loss on the batch.
        '''
        self.optimizer.zero_grad()

        self.q_model.train()

        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        # (batch, state_shape) -> (batch, num_actions)
        q_as = self.q_model(s)

        # (batch, num_actions) -> (batch, )
        Q = torch.gather(q_as, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)

        # update model
        batch_loss = self.loss(Q, y)
        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()

        self.q_model.eval()

        return batch_loss


class DMCAgent(nn.Module):
    def __init__(self, gamma=0.9, learning_rate=0.001, state_size=241,
                 layers=2, epsilon=0.1, alpha=0.4, n_games=5000):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_games = n_games

        self.state_size = state_size
        self.action_size = 61

        self.q_estimate = DMCEstimator(self.action_size, state_size=state_size,
                                       learning_rate=learning_rate, layers=layers)

    # doing epsilon greedy

    def step(self, state):
        ''' Takes a step following an epsilon greedy policy

        Args:
            state: a dictionary containing the information of the current state
        '''
        q = self.predict(state)
        acts = list(state['legal_actions'].keys())

        if uniform.rvs() < self.epsilon:  # this isn't random
            return acts[acts.index(np.argmax(q))]
        else:  # act randomly
            probs = np.ones(len(list(acts) - 1, dtype=float)) * \
                self.epsilon / (len(acts)-1)
            return acts[np.random.choice(np.arange(len(probs)), p=probs)]

    def eval_step(self, state):
        q = self.predict(state)
        best_act = np.argmax(q)
        return best_act

    def predict(self, state):

        q_values = self.q_estimator.predict_nograd(
            np.expand_dims(state['target'], 0))[0]
        masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float)
        legal_actions = list(state['legal_actions'].keys())
        masked_q_values[legal_actions] = q_values[legal_actions]

        return masked_q_values

    def train(self, n_games):
        for i in range(n_games):
            trajectories, payoffs = env.run()
            G_t = 0

            # train for first player only
            tot_states = len(trajectories[0])
            for t in range(tot_states):
                G_t += pow(self.gamma, tot_states - t) * payoffs[0]
                current_val = self.q_estimate.predict(trajectories[0][t])
                delta = self.alpha * (G_t - current_val)
                # this needs to change
                self.q_estimate.update(trajectories[0][t])
        # return weights
