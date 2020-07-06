import argparse
import copy
import microgridRLsimulator
from datetime import datetime

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

import params.params as params
from agents.random_agent import RandomAgent
from agents.agent_ppo import PPOAgent
from agents.dqn_agent import DQNAgent

parser = argparse.ArgumentParser()
parser.add_argument('--env', '-e', type=str, default='microgrid',
                    choices=['microgrid', 'maze-dense', 'maze-sparse'])

args = parser.parse_args()

params = copy.deepcopy(params.params)
params['env']['env'] = args.env

params['env']['case'] = 'elespino_discrete'
agent = PPOAgent(params)

agent.train()
# agent.test()
agent.store_results(path=f"results/{dt_string}/", id=agent.name,
                    render_tr_te=2)
print("End of agent's life")

