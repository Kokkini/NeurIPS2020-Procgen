import time
import gym
import numpy as np

from ray.tune import registry
from procgen.env import ENV_NAMES as VALID_ENV_NAMES
from collections import deque
import cv2
from envs.procgen_env_wrapper import ProcgenEnvWrapper
from gym.spaces import Box
from gym import Wrapper
from .utils import *

class RewardNormalizedWrapper(Wrapper):
    def __init__(self, env, queue_length, hue=False):
        super(RewardNormalizedWrapper, self).__init__(env)
        self.queue_length = queue_length
        self.hue = hue
        self.frames = deque(maxlen=queue_length)

        self.observation_space.low = self.observation_space.low.astype(np.int32)
        self.observation_space.high = self.observation_space.high.astype(np.int32)

        self.rew_normalizer = RewardNormalizer()
        low = np.concatenate([self.observation_space.low, self.observation_space.low-self.observation_space.high], axis=-1)
        high = np.concatenate([self.observation_space.high, self.observation_space.high], axis=-1)
        if hue:
            low = np.concatenate([low,self.observation_space.low[...,:1]],axis=-1)
            high = np.concatenate([high, self.observation_space.high[...,:1]], axis=-1)

        # print(low[:,:,3])
        # print(low[:,:,-1])
        # print(high)
        self.observation_space = Box(low=low, high=high, dtype=np.int32)
        

    def _get_observation(self):
        assert len(self.frames) == self.queue_length, (len(self.frames), self.queue_length)
        obs = np.concatenate([self.frames[-1], self.frames[-1] - self.frames[0]], axis=-1)
        if self.hue:
            print("shape", self.frames[-1].shape)
            hue = cv2.cvtColor(self.frames[-1].astype(np.uint8), cv2.COLOR_RGB2HSV)[...,:1].astype(np.int32) * 255 // 180
            obs = np.concatenate([obs, hue], axis=-1)
        return obs

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        normed_rew = self.rew_normalizer.normalize(np.array([reward]), np.array([False]))
        self.frames.append(observation.astype(np.int32))
        return self._get_observation(), normed_rew, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation.astype(np.int32)) for _ in range(self.queue_length)]
        return self._get_observation()

def create_my_custom_env(config):
    hue = config.pop("hue", False)
    queue_length = config.pop("queue_length", 2)
    env = ProcgenEnvWrapper(config)
    env = RewardNormalizedWrapper(env, queue_length, hue)
    return env


registry.register_env(
    "reward_normalized_wrapper", create_my_custom_env
)