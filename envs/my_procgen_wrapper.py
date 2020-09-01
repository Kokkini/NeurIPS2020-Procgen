import time
import gym
import numpy as np

from ray.tune import registry
from procgen.env import ENV_NAMES as VALID_ENV_NAMES
from collections import deque
import cv2
from procgen_env_wrapper import ProcgenEnvWrapper


class MyProcgenWrapper(Wrapper):
    def __init__(self, env, queue_length, hue=False):
        super(MyProcgenWrapper, self).__init__(env)
        self.queue_length = queue_length
        self.hue = hue
        self.frames = deque(maxlen=queue_length)

        low = np.concatenate([self.observation_space.low, self.observation_space.low-self.observation_space.high], axis=-1)
        high = np.concatenate([self.observation_space.high, self.observation_space.high], axis=-1)
        if hue:
            low = np.concatenate([low,self.observation_space.low[...,:1]],axis=-1)
            high = np.concatenate([high, self.observation_space.high[...,:1]], axis=-1)

        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def _get_observation(self):
        assert len(self.frames) == self.queue_length, (len(self.frames), self.queue_length)
        obs = np.concatenate([self.frames[-1], self.frames[-1] - self.frames[0]], axis=-1)
        if self.hue:
            hue = cv2.cvtColor(self.frames[-1], cv2.COLOR_RGB2HSV)[...,:1]
            obs = np.concatenate([obs, hue], axis=-1)
        return obs

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self._get_observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.queue_length)]
        return self._get_observation()

def create_my_custom_env(config):
    hue = config.pop("hue", False)
    queue_length = config.pop("queue_length", 2)
    env = ProcgenEnvWrapper(config)
    env = MyProcgenWrapper(env, queue_length, hue)
    return env


registry.register_env(
    "my_procgen_wrapper", create_my_custom_env
)