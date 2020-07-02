from agents.abstract_agent import AbstractAgent
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

class PPOAgent(AbstractAgent):
    def __init__(self, params):
        super(PPOAgent, self).__init__(params)
        self.model = PPO2(MlpPolicy, self.env, verbose=1)
        self.name = "ppo"

    def run_lifetime(self):
        self.model.learn(total_timesteps=self.params['problem']['nb_train_steps'])
        obs = self.env.reset()
        for i in range(self.params['problem']['nb_test_steps']):
                action, _states = self.model.predict(obs)
                obs, reward, dones, info = self.env.step(action)

        self.store_results(id=self.name)

