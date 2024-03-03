import numpy as np
from typing import Tuple
from copy import deepcopy

from grid import Grid
from pomdp import POMDPModel

"""
* Init the belief with uniform distribution and then it will be updated
* A belief is probability distribution over belief b
* reward R(b, a)
"""

class Node(object):
    def __init__(self, value: float):
        self.value = value
        self.nb_visits = 0
        self.children = []

    def add_child(self, child):
        self.children.append(child)

class SearchTree(object):
    def __init__(self, root: Node) -> None:
        self.root = root

class POUCTAgent(object):
    """
    """
    def __init__(self, env: Grid, pomdp_model: POMDPModel):
        """
        * The initial belief is uniform.
        * Note that there is no x and y because
        the agent does not know where it is.
        * It has just a belief over states.
        """
        self.env = env
        self.pomdp_model = pomdp_model
        self.transition_model = self.pomdp_model.get_transition_model
        self.observation_model = self.pomdp_model.get_observation_model
        # belief starts as a uniform distribution over states
        self.previous_belief = None
        self.current_belief = [
            1/self.env.get_number_states for _ in range(self.env.get_number_states)
        ]

        self.possible_actions = self.env.possible_actions

        # Sequence of tuples (action, observation)
        self.history = []

        self.search_tree = SearchTree()

    def action_str2int(self, action: str) -> int:
        return self.possible_actions[action]

    def action_int2str(self, action: int) -> str:
        for str_action in self.possible_actions.keys():
            if self.possible_actions[str_action] == action:
                return str_action

    def update_belief(self, action: str, observation: Tuple):
        """
        Bayes theorem update
        SE: state estimator
        """
        self.previous_belief = deepcopy(self.current_belief)
        self.history.append((action, observation))
        action = self.action_str2int(action)
        for state_i in range(self.env.get_number_states):
            numerator = sum(
                self.observation_model.get_observation_prob(action, state_i, observation)*
                self.transition_model.get_transition_prob(state_j, action, state_i)*
                self.previous_belief[state_i]
                for state_j in range(self.pomdp_model.get_nb_states)
            )
            denominator = sum(
                self.previous_belief[state_j]*
                sum(
                    self.transition_model.get_transition_prob(state_j, action, state_k)*
                    self.observation_model.get_observation_prob(action, state_k, observation)
                    for state_k in range(self.env.get_number_states)
                )
                for state_j in range(self.pomdp_model.get_nb_states)
            )
            self.current_belief[state_i] = numerator/denominator

    def search(self, state: int, depth: int):
        pass

    def simulate(self):
        pass

    def rollout_policy(self, state: int, depth: int):
        pass

    def learn(self):
        pass

    def take_action(self):
        """
        UCT algorithm
        """
        pass
