"""Add delay to gripper action."""

from collections import deque
from typing import Any

import gymnasium as gym


class GripperDelay(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Domain randomisation."""

    def __init__(self, env: gym.Env, buffer_size: int = 2):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        self.is_vector_env = getattr(env, "is_vector_env", False)

        self.buffer_size = buffer_size
        self.gripper_buffer = deque([1.0] * buffer_size, maxlen=buffer_size)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Any, dict]:

        self.gripper_buffer.clear()  # Clear existing buffer
        self.gripper_buffer.extend([1.0] * self.buffer_size)  # Re-populate with zeros

        return super().reset()

    def step(self, action):
        """Step and add noise to observation."""

        delayed_action = self.gripper_buffer.popleft()
        self.gripper_buffer.append(action[-1])
        action[-1] = delayed_action

        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info
