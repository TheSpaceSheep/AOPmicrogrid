import json

class AbstractAgent():
    def __init__(self, params):
        self.params = params
        self.T = params['problem']['T']

        if self.params['env']['env'] == 'microgrid':
            import gym
            import microgridRLsimulator
            from stable_baselines.common import make_vec_env
            self.env = gym.make("microgridRLsimulator-v0",
                               start_date = self.params['env']['tr_st_date'],
                               end_date = self.params['env']['tr_en_date'],
                               case = self.params['env']['case'])

            self.env = make_vec_env(lambda :self.env, n_envs=1)

        self.params['env']['obs_shape'] = self.env.observation_space.shape[0]
        self.params['env']['act_shape'] = self.env.action_space.shape[0]
        self.params['env']['min_act'] = self.env.action_space.low[0]
        self.params['env']['max_act'] = self.env.action_space.high[0]

        self.nb_timesteps = 0

    def step(self, action):
        # Execute an action in environment
        obs, rew, done, info = self.env.step(action)

        if done:
            self.prev_obs = self.env.reset()
        else:
            self.prev_obs = obs

        self.nb_timesteps += 1

        return obs, rew, done, info

    def run_lifetime(self):
        aborted = False
        try:
            while self.time < self.T:
                self.run_timestep()
        except KeyboardInterrupt:
            print('Terminating Agent')
            aborted = True
            self.env.close()

        return aborted

    def run_timestep(self):
        action = self.get_action()
        self.step(action)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs, rew, done, info

