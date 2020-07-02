import argparse
import copy
import microgridRLsimulator

import params.env_params as env_params
import params.gen_params as gen_params
from agents.random_agent import RandomAgent
from agents.agent_ppo import PPOAgent
from agents.dqn_agent import DQNAgent

parser = argparse.ArgumentParser()
parser.add_argument('--env', '-e', type=str, default='microgrid',
                    choices=['microgrid', 'maze-dense', 'maze-sparse'])

args = parser.parse_args()

params = copy.deepcopy(env_params.env_params)
params.update(gen_params.gen_params)
params['env']['env'] = args.env

params['env']['case'] = 'elespino_discrete'
agent = DQNAgent(params)
env = agent.env
env.reset()

agent.run_lifetime()
print("End of agent's life")
