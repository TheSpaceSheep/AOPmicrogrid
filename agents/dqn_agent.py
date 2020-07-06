from agents.stable_baseline_agent import StableBaselineAgent
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN

class DQNAgent(StableBaselineAgent):
    def __init__(self, params):
        super(DQNAgent, self).__init__(params)
        self.model = DQN(MlpPolicy,
                         self.env,
                         verbose=1,
                         gamma=params['agent']['dqn']['gamma'],
                         learning_rate=params['agent']['dqn']['learning_rate'],
                         buffer_size=params['agent']['dqn']['buffer_size'],
                         exploration_fraction=params['agent']['dqn']['exploration_fraction'],
                         train_freq=params['agent']['dqn']['train_freq'],
                         batch_size=params['agent']['dqn']['batch_size'],
                         double_q=params['agent']['dqn']['double_q'],
                         learning_starts=params['agent']['dqn']['learning_starts'],
                         target_network_update_freq=params['agent']['dqn']['target_network_update_freq'],
                         prioritized_replay=params['agent']['dqn']['prioritized_replay'],
                         tensorboard_log=params['agent']['dqn']['tensorboard'])

        self.name = "dqn"

