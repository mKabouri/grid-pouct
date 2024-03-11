import time
import numpy as np
from typing import Tuple, List
from copy import deepcopy

from grid import Grid
from pomdp import POMDPModel

"""
* A belief is probability distribution over belief b
* reward R(b, a)
"""

class Node(object):
    def __init__(
        self,
        action: str=None, # None for none action nodes
        parent=None,
        value: float=0.,
        nb_visits: int=0,
        history: List=[],
        children: List=[]
    ):
        self.action = action
        self.parent = parent
        self.value = value
        self.nb_visits = nb_visits
        self.history = history
        self.children = children
        # self.belief

    def __str__(self) -> str:
        return f"Node({self.nb_visits}, {self.value}, {len(self.history)}, {len(self.children)})"

    def __repr__(self) -> str:
        return self.__str__()

    def add_child(self, child):
        self.children.append(child)

    @property
    def is_leaf(self):
        return self.children == []

    @property
    def get_children(self):
        return self.children

    @property
    def get_history(self):
        return self.history

    @property
    def get_nb_visits(self):
        return self.nb_visits

class SearchTree(object):
    def __init__(self, root: Node) -> None:
        self.root = root
        self.current_node = root

    def is_in_tree(self, history: List) -> bool:
        def is_in_tree_from_node(node: Node) -> bool:    
            if node.get_history == history:
                return True
            for child in node.get_children:
                if is_in_tree_from_node(child):
                    return True
            return False

        if self.root.get_history == history:
            return True
        for child in self.root.get_children:
            if is_in_tree_from_node(child):
                return True
        return False

    @property
    def get_root(self):
        return self.root

    @property
    def get_current_node(self):
        return self.current_node

class POMCPAgent(object):
    """
    """
    def __init__(
        self,
        env: Grid,
        pomdp_model: POMDPModel,
        generator,
        initial_state: Node=Node(),
        time_out: float=30,
        discount_factor: int=1,
        init_belief: List=None,
        ucb_cst:float=np.sqrt(2),
    ):
        """
        * The initial belief is uniform.
        * Note that there is no x and y because
        the agent does not know where it is.
        * It has just a belief over states.
        """
        self.env = env
        self.pomdp_model = pomdp_model
        self.discout_factor = discount_factor
        self.time_out = time_out
        self.ucb_cst = ucb_cst

        self.transition_model = self.pomdp_model.get_transition_model
        self.observation_model = self.pomdp_model.get_observation_model
        # belief starts as a uniform distribution over states
        self.previous_belief = None
        if init_belief:
            self.current_belief = init_belief
        else:
            self.current_belief = [
                1/self.env.get_number_states for _ in range(self.env.get_number_states)
            ]

        self.possible_actions = self.env.possible_actions

        # Sequence of tuples (action, observation)
        # The agent own history
        self.history = []
        self.generator = generator
        self.search_tree = SearchTree(initial_state)

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

    def sample_state_from_belief(self):
        return np.random.choice(len(self.current_belief), p=self.current_belief)

    def search(self):
        start_time = time.time()
        while time.time() - start_time < self.time_out:
            state = self.sample_state_from_belief()
            print(f"initial state: {state}")
            self.simulate(state, self.history, depth=5)
        # Values over the actions
        children_values = [
            child.value
            for child in self.search_tree.get_root.get_children
        ]
        return np.argmax(children_values)

    def simulate(self, state: int, history: List, depth: int):
        current_node = self.search_tree.get_current_node

        if self.discout_factor**depth < 0.005:
            return 0

        if not self.search_tree.is_in_tree(history):
            for action in self.possible_actions.keys():
                current_node.add_child(Node(parent=current_node,action=action,
                                            history=deepcopy(history).append(action)))
            return self.rollout_policy(state, depth)

        values = [
            child.value + self.ucb_cst*np.sqrt(np.log(current_node.get_nb_visits)/child.get_nb_visits)
            for child in current_node.children
        ]
        best_index = np.argmax(values)
        best_child = current_node.children[best_index]
        best_action = best_child.history[-1]
        next_state, next_obs, reward = self.generator(state, best_action)
        # Update the belief
        self.update_belief(best_action, next_obs)

        new_history = deepcopy(history)
        new_history.append(best_action)
        new_history.append(next_obs)
        ret = reward + self.discout_factor*self.simulate(next_state,
                                                         new_history,
                                                         depth+1)
        current_node.nb_visits += 1
        best_child.nb_visits += 1
        best_child.value += (ret-best_child.value)/best_child.nb_visits

        return ret

    def rollout_policy(self):
        """
        Returns an action randomly (random policy)
        """
        return self.action_int2str(np.random.choice(len(self.possible_actions)))

    def rollout(self, state: int, depth: int):
        if self.discout_factor**depth < 0.005:
            return 0
        action = self.rollout_policy()
        next_state, _, reward = self.generator(state, action)
        return reward + self.discout_factor*self.rollout(next_state, depth+1)

    def _plan(self):
        pass

    def take_action(self):
        return self._plan()
