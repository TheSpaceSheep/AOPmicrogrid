from agents.abstract_agent import AbstractAgent
import numpy as np

class RandomAgent(AbstractAgent):
    """ an agent that takes random continuous actions in the  microgrid
    simulator """
    def __init__(self, params):
        super(RandomAgent, self).__init__(params)

    def get_action(self):
        action = self.env.action_space.sample()
        return action
