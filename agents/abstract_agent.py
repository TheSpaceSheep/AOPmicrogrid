class AbstractAgent():
    def __init__(self, params):
        self.params = params
        self.T = params['problem']['T']

        if self.params['env']['env'] == 'microgrid':
            import gym
            import microgridRLsimulator
            self.env = gym.make("microgridRLsimulator-v0",
                               start_date = self.params['env']['tr_st_date'],
                               end_date = self.params['env']['tr_en_date'],
                               case = self.params['env']['case'])

            self.test_env = gym.make("microgridRLsimulator-v0",
                               start_date = self.params['env']['te_st_date'],
                               end_date = self.params['env']['te_en_date'],
                               case = self.params['env']['case'])

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



    def train(self):
        self.model.learn(total_timesteps=self.params['problem']['nb_train_steps'])

    def test(self):
        obs = self.test_env.reset()
        for i in range(self.params['problem']['nb_test_steps']):
                action, _states = self.model.predict(obs)
                obs, reward, dones, info = self.test_env.step(action)


    def store_results(self, path="../plots/", id=None, render_tests=False):
        if render_tests:
            self.test_env.render(path+"test/", id=id)
        else:
            self.env.render(path+"train/", id=id)

