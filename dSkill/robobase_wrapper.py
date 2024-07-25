"""Wrapper for robobase which includes env factory."""

import gymnasium as gym
import hydra
from gymnasium.wrappers import TimeLimit
from omegaconf import DictConfig
from robobase.envs.env import EnvFactory
from robobase.envs.wrappers import (
    ActionSequence,
    AppendDemoInfo,
    AppendIsFirst,
    ConcatDim,
    FrameStack,
    OnehotTime,
    RescaleFromTanh,
)

from dSkill.real.envs.wrappers.sim_to_real import Sim2Real
from dSkill.sim.envs.wrappers.aruco_rand import ArucoRand


class dSkillFactory(EnvFactory):
    """Creates the dex manip environments for robobase."""

    def __init__(self, sim=True):
        super().__init__()
        self.sim = sim

    def _wrap_env(self, env, cfg):

        if self.sim and cfg.env.name == "vertical_slide":
            if not cfg.pixels:
                env = ArucoRand(env)
        elif not self.sim:
            env = Sim2Real(env)

        env = RescaleFromTanh(env)
        env = ConcatDim(env, 1, -1, "low_dim_state")
        env = TimeLimit(env, cfg.env.episode_length)
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(env, cfg.env.episode_length)
        env = FrameStack(env, cfg.frame_stack)
        env = ActionSequence(env, cfg.action_sequence)
        env = AppendDemoInfo(env)
        if cfg.replay.sequential:
            env = AppendIsFirst(env)

        return env

    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:  # dead: disable
        """Creates the vectorised (wrapped) gym environment.

        Args:
            cfg: The hydra configuration.

        Returns: The vectorised gym environment.
        """
        vec_env_class = gym.vector.AsyncVectorEnv
        return vec_env_class(
            [
                lambda: self._wrap_env(
                    hydra.utils.instantiate(cfg.env.create),
                    cfg,
                )
                for _ in range(cfg.num_train_envs)
            ],
            context="spawn",
        )

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        """Creates a standard (wrapped) gym environment for evaluation.

        Args:
            cfg: The hydra configuration.

        Returns: A non-vectorised gym environment.
        """
        return self._wrap_env(
            hydra.utils.instantiate(cfg.env.create),
            cfg,
        )
