from agents.abstract_agent import AbstractAgent
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

class PPOAgent(AbstractAgent):
    def __init__(self, params):
        super(PPOAgent, self).__init__(params)
        self.model = PPO2(MlpPolicy, self.env, verbose=1)
        self.name = "ppo"

