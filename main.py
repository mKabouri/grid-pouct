from grid import make_env
from agent import POMCPAgent
from pomdp import make_pomdp_model
from generator import Generator

if __name__ == "__main__":
    print("Start")
    pomdp_model = make_pomdp_model()
    env = make_env(pomdp_model)
    obs = env.reset()
    print(obs)
    agent = POMCPAgent(env)
    print("Planning starts")
    action = agent.take_action()
    print(f"Action planned: {action}")
