from agents.stable_baseline_agent import StableBaselineAgent
from stable_baselines.common.policies import MlpPolicy, CnnPolicy, LstmPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

class PPOAgent(StableBaselineAgent):
    def __init__(self, params):
        super(PPOAgent, self).__init__(params)
        if params['agent']['ppo']['policy'] == 'MlpPolicy':
            from stable_baselines.common.policies import MlpPolicy as Policy
        elif params['agent']['ppo']['policy'] == 'LstmPolicy':
            from stable_baselines.common.policies import LstmPolicy as Policy

        self.model = PPO2(Policy,
                          self.env,
                          verbose=1,
                          gamma=params['agent']['ppo']['gamma'],
                          n_steps=params['agent']['ppo']['n_steps'],
                          ent_coef=params['agent']['ppo']['ent_coef'],
                          learning_rate=params['agent']['ppo']['learning_rate'],
                          vf_coef=params['agent']['ppo']['vf_coef'],
                          max_grad_norm=params['agent']['ppo']['max_grad_norm'],
                          lam=params['agent']['ppo']['lam'],
                          nminibatches=params['agent']['ppo']['nminibatches'],
                          noptepochs=params['agent']['ppo']['noptepochs'],
                          cliprange=params['agent']['ppo']['cliprange'],
                          tensorboard_log=params['agent']['ppo']['tensorboard'])

        self.name = "ppo"

