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
from envs.utils import RewardNormalizer

class StackAndSubtract(Wrapper):
    def __init__(self, env, queue_length, reward_norm=False, death_penalty=None):
        super(StackAndSubtract, self).__init__(env)
        self.queue_length = queue_length
        self.frames = deque(maxlen=queue_length)
        self.rew_normalizer = None
        self.death_penalty = death_penalty
        if reward_norm:
            self.rew_normalizer = RewardNormalizer()

        high = np.concatenate([self.observation_space.high]*(queue_length+1), axis=-1)
        low = np.concatenate([self.observation_space.low]*(queue_length+1), axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=np.uint8)
        

    def _get_observation(self):
        assert len(self.frames) == self.queue_length, (len(self.frames), self.queue_length)
        subtract = (self.frames[-1] - self.frames[0] + 255) // 2
        obs = np.concatenate(self.frames +[subtract], axis=-1)
        obs = obs.astype(np.uint8)
        return obs

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation.astype(np.int32))
        if self.death_penalty is not None:
            if done and reward == 0:
                reward = self.death_penalty
        if self.rew_normalizer is not None:
            reward = self.rew_normalizer.normalize(np.array([reward]), np.array([False]))[0]
        return self._get_observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation.astype(np.int32)) for _ in range(self.queue_length)]
        return self._get_observation()

def create_my_custom_env(config):
    queue_length = int(config.pop("queue_length", 2))
    reward_norm = config.pop("reward_norm", False)
    death_penalty = config.pop("death_penalty", None)
    if death_penalty is not None:
        death_penalty = float(death_penalty)
    env = ProcgenEnvWrapper(config)
    env = StackAndSubtract(env, queue_length, reward_norm, death_penalty)
    return env


registry.register_env(
    "stack_and_subtract", create_my_custom_env
)