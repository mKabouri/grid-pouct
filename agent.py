import numpy as np
from typing import Tuple

from grid import Grid

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
    def __init__(self, env: Grid) -> None:
        """
        * Note that there is no x and y because
        the agent does not know where it is.
        * It has just a belief over states.
        """
        self.env = env
        # belief starts as a uniform distribution over states
        self.belief = [
            1/self.env.get_number_states for _ in range(self.env.get_number_states)
        ]

        self.possible_actions = self.env.possible_actions

        # Sequence of tuples (action, observation)
        self.history = []

        self.search_tree = SearchTree()

    def update_belief(self, action: str, observation: Tuple):
        """
        I will use bayes theorem update
        SE: state estimator
        """
        self.history.append((action, observation))
        for i in range(self.env.get_number_states):
            # self.belief[i] = 
            pass

    def search(self, state: int, depth: int):
        pass

    def simulate(self):
        pass

    def rollout_policy(self, state: int, depth: int):
        pass

    def learn(self):
        pass

    def take_action(self):
        pass
