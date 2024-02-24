import numpy as np
from grid import Grid

class POUCTAgent(object):
    def __init__(self, env: Grid) -> None:
        # Note that there is no x and y because
        # the agent does not know where it is
        # It has just a belief over states
        self.env = env
        # belief start as a uniform distribution over states
        self.belief = [
            1/self.env.get_number_states for _ in range(self.env.get_number_states)
        ]

        self.possible_actions = self.env.possible_actions

    def update_belief(self):
        """
        SE: state estimator
        """
        pass

    def take_action(self):
        pass

    def learn(self):
        pass
