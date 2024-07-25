"""Domain randomisation."""

import gymnasium as gym
import numpy as np


class ArucoRand(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Domain randomisation."""

    def __init__(self, env: gym.Env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def observation(self, observation):
        """Adds noise to observation."""
        offset_rng = 0.005

        random_offset = np.array(
            [
                np.random.uniform(-offset_rng, offset_rng),
                np.random.uniform(-offset_rng, offset_rng),
                np.random.uniform(-offset_rng, offset_rng),
            ],
        )
        observation["p_m1_in_m0"] += random_offset

        random_offset = np.array(
            [
                np.random.uniform(-offset_rng, offset_rng),
                np.random.uniform(-offset_rng, offset_rng),
                np.random.uniform(-offset_rng, offset_rng),
            ],
        )
        observation["p_m2_in_m0"] += random_offset

        return observation

    def step(self, action):
        """Step and add noise to observation."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, truncated, info
