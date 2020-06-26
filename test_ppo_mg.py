from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from envs.ContinualParticleMaze import ContinualParticleMaze
import time
import gym
import microgridRLsimulator

env = gym.make("microgridRLsimulator-v0",
               start_date="2016-01-01",
               end_date="2017-07-31",
               case="elespino")

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000)

obs = env.reset()

for i in range(10000):
    action, _states = model.predict(obs)
    obs, reward, dones, info = env.step(action)
    time.sleep(0.1)

env.render("plots/")
