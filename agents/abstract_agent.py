class AbstractAgent():
    def __init__(self, params):
        self.params = params
        self.T = params['problem']['T']

        if self.params['env']['env'] == 'microgrid':
            import gym
            import microgridRLsimulator
            self.env = gym.make("microgridRLsimulator-v0",
                               start_date = self.params['env']['start_date'],
                               end_date = self.params['env']['end_date'],
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

    def get_action(self):
        raise NotImplementedError

    def run_timestep(self):
        action = self.get_action()
        print("Taking action", action)
        self.step(action)

    def run_lifetime(self):
        aborted = False
        try:
            while self.nb_timesteps < self.T:
                self.run_timestep()
        except KeyboardInterrupt:
            print('Terminating agent early')
            aborted = True

        self.env.close()
        return aborted

    def store_results(self, path="../plots/", id=None):
        self.env.render(path, id=id)

