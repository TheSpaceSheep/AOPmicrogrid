from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from envs.ContinualParticleMaze import ContinualParticleMaze
import time
import gym
import microgridRLsimulator

env = ContinualParticleMaze(grid_name="8", dense=True)

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
env.render()

while True:
    action, _states = model.predict(obs)
    obs, reward, dones, info = env.step(action)
    env.render()
    time.sleep(0.1)

