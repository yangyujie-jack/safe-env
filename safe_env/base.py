import gym


class BarrierEnv(gym.Env):
    @property
    def barrier_input_dim(self):
        return self.observation_space.shape[0]

    @staticmethod
    def preprocess(obs):
        return obs

    @staticmethod
    def handcraft_barrier(obs):
        raise NotImplementedError
