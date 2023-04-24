import numpy as np
import torch
from collections import OrderedDict
from icecream import ic

# LOCAL IMPORT
from uno.envs.env import Env
from uno.game.uno import Game
from uno.game.uno.utils import encode_hand, encode_target
from uno.game.uno.utils import ACTION_SPACE, ACTION_LIST
from uno.game.uno.utils import cards2list


BASE_ID = 0
TRAIN_ID = 1
torch.manual_seed(2023)
np.random.seed(2023)


class UnoEnv2P():

    def __init__(self, base_agent, training_agent):
        self.name = 'uno-2p'
        self.game = Game()
        # Agent declaration
        self.base_agent = base_agent
        self.training_agent = training_agent
        self.agents = [base_agent, training_agent]
        # Statistics
        self.action_recorder = []  # only record the action of training_agent
        self.states = [[4, 4, 15] for _ in range(2)]
        self.actions = [None for _ in range(2)]
        # Initialization
        _, self.cur_player = self.game.init_game()
        self.cur_state = self.get_state(self.cur_player)
        self.base_move()

    def reset(self):
        self.game = Game()
        self.action_recorder = []  # only record the action of training_agent
        # Initialization
        _, self.cur_player = self.game.init_game()
        self.cur_state = self.get_state(self.cur_player)
        self.base_move()

    def base_move(self):
        while self.cur_player == 0 and not self.game.is_over():
            base_action = self.base_agent.step(self.cur_state)
            # ic(base_action)
            if not self.base_agent.use_raw:
                base_action = self._decode_action(base_action)
            _, self.cur_player = self.game.step(base_action)
            self.cur_state = self.get_state(self.cur_player)
        # update current state
        assert self.cur_player == 1 or self.game.is_over()
        self.cur_state = self.get_state(1)

    def step(self, action):

        # Start accessing actions
        assert self.cur_player == 1
        # Perform one-step update
        # ic(action)
        if not self.training_agent.use_raw:
            action = self._decode_action(action)
        # Record the action for human agent
        self.action_recorder.append(("Training", action))
        _, self.cur_player = self.game.step(action)
        self.cur_state = self.get_state(self.cur_player)

        # Update current player
        if self.cur_player == 1:
            return self.cur_state, self.game.is_over()
        else:
            self.base_move()
            return self.cur_state, self.game.is_over()

    def get_state(self, id):
        return self._extract_state(self.game.get_state(id))

    def _extract_state(self, state):
        obs = np.zeros((4, 4, 15), dtype=int)
        encode_hand(obs[:3], state['hand'])
        encode_target(obs[3], state['target'])
        legal_action_id = self._get_legal_actions()
        extracted_state = {'obs': obs, 'legal_actions': legal_action_id}
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [
            a for a in state['legal_actions']]
        extracted_state['action_record'] = self.action_recorder
        return extracted_state

    def get_payoffs(self):

        return np.array(self.game.get_payoffs())

    def _decode_action(self, action_id):
        legal_ids = self._get_legal_actions()
        # ic(action_id)
        assert action_id in legal_ids
        if action_id in legal_ids:
            return ACTION_LIST[action_id]
        return ACTION_LIST[np.random.choice(legal_ids)]

    def _get_legal_actions(self):
        legal_actions, target = self.game.get_legal_actions()
        # ic(target.str)
        legal_ids = {ACTION_SPACE[action]: None for action in legal_actions}
        return OrderedDict(legal_ids)

    def is_over(self):
        return self.game.is_over()

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['num_players'] = self.num_players
        state['hand_cards'] = [cards2list(player.hand)
                               for player in self.game.players]
        state['played_cards'] = cards2list(self.game.round.played_cards)
        state['target'] = self.game.round.target.str
        state['current_player'] = self.game.round.current_player
        state['legal_actions'], _ = self.game.round.get_legal_actions(
            self.game.players, state['current_player'])
        return state


# unoenv = UnoEnv2P()
