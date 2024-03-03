import numpy as np

import config

"""
POMDP: <S, A, Z, T, R, O>, where:
    * S: states (defined in grid.py ???)
    * A: actions
    * Z: observables: (color of the cell?)
    * T: transition function (s, a, s')
    * R: reward function (s, a) (-1 for each step)
    * O: observation function (s', a, z)
"""

class TransitionModel:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.transition_probs = np.zeros((n_states, n_actions, n_states))

    def set_transition(self, state, action, next_state, probability):
        self.transition_probs[state, action, next_state] = probability

    def get_transition_prob(self, state, action, next_state):
        return self.transition_probs[state, action, next_state]

class ObservationModel:
    def __init__(self, n_states, n_actions, n_observations):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_observations = n_observations
        self.observation_probs = np.zeros((n_actions, n_states, n_observations))

    def set_observation(self, action, state, observation, probability):
        self.observation_probs[action, state, observation] = probability

    def get_observation_prob(self, action, state, observation):
        return self.observation_probs[action, state, observation]

class RewardModel:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.rewards = np.full((n_states, n_actions), -1)

    def set_reward(self, state, action, reward):
        self.rewards[state, action] = reward

    def get_reward(self, state, action):
        return self.rewards[state, action]

class POMDPModel:
    def __init__(self, n_states, n_actions, n_observations):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_observations = n_observations

        # Create transition, observation and reward models
        self.transition_model = TransitionModel(n_states, n_actions)
        self.observation_model = ObservationModel(n_states, n_actions, n_observations)
        self.reward_model = RewardModel(n_states, n_actions)

    @property
    def get_nb_states(self):
        return self.n_states

    @property
    def get_nb_observations(self):
        return self.n_observations

    @property
    def get_nb_actions(self):
        return self.n_actions

    @property
    def get_reward_model(self):
        return self.reward_model
        
    @property
    def get_transition_model(self):
        return self.transition_model

    @property
    def get_observation_model(self):
        return self.observation_model

def make_pomdp_model():
    # +1 in n_observations for the goal observation
    return POMDPModel(config.NB_STATES, len(config.ACTIONS), len(config.POSSIBLE_COLORS)+1)
