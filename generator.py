import numpy as np
from pomdp import POMDPModel
import config

def Generator(pomdp: POMDPModel):
    def generator(state: int, action: str):
        new_state = np.random.multinomial(1, pomdp.get_transition_model.transition_probs[:, action, state])
        new_state = int(np.nonzero(new_state)[0])
        new_obs = np.random.multinomial(1, pomdp.get_observation_model.observation_probs[action, state, :])
        new_obs = int(np.nonzero(new_obs)[0])
        reward = pomdp.get_reward_model.get_reward(state, action)
        return new_state, new_obs, reward
    return generator
