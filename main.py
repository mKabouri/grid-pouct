from grid import make_env
from agent import POMCPAgent
from pomdp import make_pomdp_model
from generator import Generator

if __name__ == "__main__":
    print("Start")
    env = make_env()
    obs = env.reset()
    print(obs)
    pomdp_model = make_pomdp_model()
    generator = Generator(pomdp_model)
    agent = POMCPAgent(env, pomdp_model, generator)
    action = agent.take_action()
    print(f"Action planned: {action}")
    reward, new_obs, done = env.step(action)
