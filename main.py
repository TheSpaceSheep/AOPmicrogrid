import argparse
import copy
import microgridRLsimulator

import params.env_params as env_params
from agents.random_agent import RandomAgent

parser = argparse.ArgumentParser()
parser.add_argument('--env', '-e', type=str, default='microgrid',
                    choices=['microgrid', 'maze-dense', 'maze-sparse'])

args = parser.parse_args()

params = copy.deepcopy(env_params.env_params)
params['env']['env'] = args.env


agent = RandomAgent(params)
env = agent.env
env.reset()

for i in range(100):
    action = agent.get_action()
    env.step(action)
    print("Taking action", action)
