import os
import sys

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = (
        "glfw" if sys.platform == "darwin" or "DISPLAY" in os.environ else "egl"
    )

# ruff: noqa: E402
import logging
from abc import abstractmethod
from collections.abc import Iterable
from enum import Enum
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from gymnasium import spaces
from mojo import Mojo

from dSkill.sim.consts import SEG_GROUP_COLORS, TARGET_SEG_GROUP, WORLD_XML
from dSkill.sim.envs.renderer import DexRenderer
from dSkill.sim.robot import Robot

# Constants
PHYSICS_DT = 0.002  # The time passed per simulation step
PHYSICS_STEPS = 10  # Controls the control frequency of the policy
PHYSICS_STEPS_GRIPPER_REFLEX = 10
RENDER_PREVIEW_SIZE = 256
RENDER_STEP_DIVIDER = 1
RENDER_PREVIEW_CAMERA_NAME = "third_person_camera"
ACTION_DELTA = 4.0

GRIPPER_REFLEX_COOLDOWN = 5  # How many steps should the gripper flex command take
GRIPPER_OPEN_SETPOINT = 0
GRIPPER_CLOSE_SETPOINT = 255


class ActionMode(Enum):
    ABS = "Absolute"
    DELTA = "Delta"


log = logging.getLogger(__name__)


class dSkillGymEnv(gym.Env):
    """Core dSkill environment which loads in common robot across all tasks."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 500,
    }

    def __init__(
        self,
        use_gripper_reflex: bool = True,
        use_tcp_in_obs: bool = False,
        use_gripper_joints_in_obs: bool = False,
        cameras: list[str] = None,
        camera_resolution: tuple[int, int] | list[int] = None,
        render_mode: str | None = None,
        pixels: bool = False,
        pixels_rgb: bool = False,
        pixels_segment: bool = False,
        pixels_masked: bool = False,
        pixels_diff: bool = False,
        start_seed: int | None = None,
    ):
        if start_seed is None:
            start_seed = np.random.randint(2**32)
        if not isinstance(start_seed, int):
            raise ValueError("Expected start_seed to be an integer.")
        self._next_seed = start_seed  # Seed that will be used on next reset()
        self._current_seed = None  # Seed that was used in last reset()

        self.cameras = cameras or []
        self.camera_resolution = (
            [640, 480] if camera_resolution is None else camera_resolution
        )
        self.use_gripper_reflex = use_gripper_reflex
        self.use_tcp_in_obs = use_tcp_in_obs
        self.use_gripper_joints_in_obs = use_gripper_joints_in_obs
        self.pixels = pixels
        self.pixels_rgb = pixels_rgb
        self.pixels_segment = pixels_segment
        self.pixels_masked = pixels_masked
        self.pixels_diff = pixels_diff
        self.prev_pixels = {}
        self.segment_groups = {}

        if self.pixels:
            assert (
                self.pixels_rgb
                or self.pixels_segment
                or self.pixels_masked
                or self.pixels_diff
            ), log.warning(
                "If pixels is True, at least one of pixels_rgb, pixels_segment, pixels_masked, or pixels_diff must be True"
            )

        log.info(f"Policy control frequency: {(1.0 / PHYSICS_DT) / PHYSICS_STEPS} Hz.")
        log.info(f"Use gripper reflex: {self.use_gripper_reflex}.")

        assert self.metadata["render_modes"] == [
            "human",
            "rgb_array",
            "depth_array",
        ], self.metadata["render_modes"]

        self.render_mode = render_mode

        self._mojo = Mojo(WORLD_XML, timestep=PHYSICS_DT)
        self._robot = Robot(self._mojo)

        self.initialise_env()
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.mujoco_renderer: DexRenderer | None = None
        self.obs_renderer: mujoco.Renderer | None = None
        self.preview_renderer: mujoco.Renderer | None = None
        self._initialize_renderers()
        self._env_steps_this_episode = 0
        self._gripper_reflex_triggered = False

        # Used for "human" render mode
        self._render_fig = self._render_ax = self._render_im = None

    def initialise_env(self):
        """Can be overwritten to add task specific items to scene."""

    def _initialize_renderers(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
            self.mujoco_renderer = None
        if self.obs_renderer is not None:
            self.obs_renderer.close()
            self.obs_renderer = None
        if self.preview_renderer is not None:
            self.preview_renderer.close()
            self.preview_renderer = None
        if self.render_mode == "human":
            self.mujoco_renderer: DexRenderer = DexRenderer(self._mojo)
        if len(self.cameras) > 0:
            h, w = self._get_camera_resolution()
            self.obs_renderer = mujoco.Renderer(self._mojo.model, h, w)

    def _get_camera_resolution(self):
        """Determines camera resolution based on the given settings."""
        h, w = (
            self.camera_resolution
            if isinstance(self.camera_resolution, Iterable)
            else (self.camera_resolution, self.camera_resolution)
        )
        return h, w

    def _get_camera_observation_elements(self, camera_name):
        space_dict = {}
        if self.pixels_rgb:
            space_dict[f"{camera_name}_rgb"] = spaces.Box(
                0,
                255,
                shape=(3, self.camera_resolution[0], self.camera_resolution[1]),
                dtype=np.uint8,
            )
        if self.pixels_segment:
            space_dict[f"{camera_name}_seg_rgb"] = spaces.Box(
                0,
                255,
                shape=(3, self.camera_resolution[0], self.camera_resolution[1]),
                dtype=np.uint8,
            )
        if self.pixels_masked:
            space_dict[f"{camera_name}_masked_rgb"] = spaces.Box(
                0,
                255,
                shape=(3, self.camera_resolution[0], self.camera_resolution[1]),
                dtype=np.uint8,
            )
        if self.pixels_diff:
            space_dict[f"{camera_name}_diff_rgb"] = spaces.Box(
                0,
                255,
                shape=(3, self.camera_resolution[0], self.camera_resolution[1]),
                dtype=np.uint8,
            )
        return space_dict

    def set_segmentation_group(self, group_name, geom_ids):
        self.segment_groups[group_name] = geom_ids

    def _get_pixels_segmentation(self, camera_name):
        pixels_geom_id = self.render(camera_name, self.camera_resolution, segment=True)[
            :, :, 0
        ]
        pixels_geom_id = pixels_geom_id.reshape(*pixels_geom_id.shape, 1).astype(
            np.uint8
        )

        width, height = self.camera_resolution
        pixels_segment = np.zeros((height, width, 3), dtype=np.uint8)

        for key, geom_ids in self.segment_groups.items():
            mask = np.isin(pixels_geom_id[:, :, 0], geom_ids)
            pixels_segment[mask] = SEG_GROUP_COLORS[key]

        return pixels_segment

    def _get_pixels_masked(self, pixels, pixels_segment, masked_group_id):
        width, height = self.camera_resolution
        pixels_masked = np.zeros((height, width, 3), dtype=np.uint8)

        mask = np.all(pixels_segment == SEG_GROUP_COLORS[masked_group_id], axis=-1)
        pixels_masked[mask] = pixels[mask]

        return pixels_masked

    def _get_pixels_diff(self, pixels, prev_pixels):
        diff = np.abs(pixels - prev_pixels)
        return diff

    def _get_masked_observation(self, pixels, pixels_segment):
        if self.pixels_segment:
            pixels = pixels_segment
        return self._get_pixels_masked(pixels, pixels_segment, TARGET_SEG_GROUP)

    def _get_diff_observation(self, scoped_camera_name, pixels_segment):
        pixels = self._get_pixels_masked(
            pixels_segment, pixels_segment, TARGET_SEG_GROUP
        )

        cached_prev_pixels = self.prev_pixels.get(scoped_camera_name)
        prev_pixels = cached_prev_pixels if cached_prev_pixels is not None else pixels
        self.prev_pixels[scoped_camera_name] = pixels
        return self._get_pixels_diff(pixels, prev_pixels)

    def _get_camera_observation(self, camera_name):
        obs_dict = {}
        scoped_camera_name = self._robot.get_scoped_camera_name(camera_name)
        pixels = self.render(scoped_camera_name, self.camera_resolution)

        needs_segmentation = (
            self.pixels_segment or self.pixels_masked or self.pixels_diff
        )
        pixels_segment = (
            self._get_pixels_segmentation(scoped_camera_name)
            if needs_segmentation
            else None
        )

        if self.pixels_rgb:
            obs_dict[f"{camera_name}_rgb"] = pixels
            obs_dict[f"{camera_name}_rgb"] = np.transpose(
                obs_dict[f"{camera_name}_rgb"], (2, 0, 1)
            )

        if self.pixels_segment:
            obs_dict[f"{camera_name}_seg_rgb"] = pixels_segment
            obs_dict[f"{camera_name}_seg_rgb"] = np.transpose(
                obs_dict[f"{camera_name}_seg_rgb"], (2, 0, 1)
            )

        if self.pixels_masked:
            obs_dict[f"{camera_name}_masked_rgb"] = self._get_masked_observation(
                pixels, pixels_segment
            )
            obs_dict[f"{camera_name}_masked_rgb"] = np.transpose(
                obs_dict[f"{camera_name}_masked_rgb"], (2, 0, 1)
            )

        if self.pixels_diff:
            obs_dict[f"{camera_name}_diff_rgb"] = self._get_diff_observation(
                scoped_camera_name, pixels_segment
            )
            obs_dict[f"{camera_name}_diff_rgb"] = np.transpose(
                obs_dict[f"{camera_name}_diff_rgb"], (2, 0, 1)
            )

        return obs_dict

    def get_observation_space(self) -> spaces.Space:
        """Create the observation space."""

        if self.use_gripper_joints_in_obs:
            n_joints = 14
        else:
            n_joints = 6

        internal_dict = {
            "proprioception": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_joints,),
                dtype=np.float32,
            ),
            "gripper_reflex": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        }

        if self.use_tcp_in_obs:
            internal_dict.update(
                {
                    "tcp_position": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(3,),
                        dtype=np.float32,
                    )
                }
            )

        if self.pixels:
            for camera_name in self.cameras:
                internal_dict.update(self._get_camera_observation_elements(camera_name))

        return spaces.Dict(internal_dict)

    def get_action_space(self) -> spaces.Space:
        low = np.array(
            [
                -1 * np.deg2rad(ACTION_DELTA),
                -1 * np.deg2rad(ACTION_DELTA),
                -1 * np.deg2rad(ACTION_DELTA),
                0,
            ],
        )
        high = np.array(
            [
                np.deg2rad(ACTION_DELTA),
                np.deg2rad(ACTION_DELTA),
                np.deg2rad(ACTION_DELTA),
                1,
            ],
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_proprioception(self):

        if self.use_gripper_joints_in_obs:
            all_joints = (
                self._robot._robot_arm_joints + self._robot._robot_gripper_joints
            )
        else:
            all_joints = self._robot._robot_arm_joints

        return np.array([joint.get_joint_position() for joint in all_joints])

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get the observations that are common to all envs."""
        obs = {
            "proprioception": self._get_proprioception().astype(np.float32),
            "gripper_reflex": np.array(
                [self._gripper_reflex_triggered],
            ).astype(np.float32),
        }

        if self.use_tcp_in_obs:
            obs.update(
                {
                    "tcp_position": self._robot.tcp_pose[:3].astype(np.float32),
                }
            )

        if self.pixels:
            for camera_name in self.cameras:
                obs.update(self._get_camera_observation(camera_name))

        return obs

    def _update_seed(self, override_seed=None):
        """Update the seed for the environment.

        Args:
            override_seed: If not None, the next seed will be set to this value.
        """
        if override_seed is not None:
            if not isinstance(override_seed, int):
                logging.warning(
                    "Expected override_seed to be an integer. Casting to int.",
                )
                override_seed = int(override_seed)
            self._next_seed = override_seed
        self._current_seed = self._next_seed
        assert self._current_seed is not None
        self._next_seed = np.random.randint(2**32)
        np.random.seed(self._current_seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Any, dict]:
        """Reset the environment.

        Args:
           seed: If not None, the environment will be reset with this seed.
           options: Additional information to specify how the environment is reset
            (optional, depending on the specific environment).
        """

        self._env_steps_this_episode = 0
        self._gripper_reflex_action_cooldown = 0
        self._update_seed(override_seed=seed)
        mujoco.mj_resetData(self._mojo.model, self._mojo.data)
        self._robot.reset()
        self.on_reset()
        self._initialize_renderers()

        return self.get_observation(), self.get_info()

    def on_reset(self):
        """Can be used to add extra objects to scene."""

    def _determine_gripper_action(self, gripper_input: float) -> float:
        """Determine and set the gripper action based on the input."""
        self._gripper_reflex_triggered = gripper_input < 0.5
        return (
            GRIPPER_OPEN_SETPOINT
            if self._gripper_reflex_triggered
            else GRIPPER_CLOSE_SETPOINT
        )

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        """Step the puck environment.

        Children should implement calculate_reward() and should_terminate().
        """

        action[-1] = self._determine_gripper_action(action[-1])

        self._robot.control(action)

        if self._gripper_reflex_triggered and self.use_gripper_reflex:

            ctrl_indices = self._robot.gripper_joints_ctrl_idxs
            for _ in range(GRIPPER_REFLEX_COOLDOWN):
                self._mojo.step()
            self._mojo.data.ctrl[ctrl_indices] = GRIPPER_CLOSE_SETPOINT
            for _ in range(GRIPPER_REFLEX_COOLDOWN, PHYSICS_STEPS_GRIPPER_REFLEX):
                self._mojo.step()
        else:
            for _ in range(PHYSICS_STEPS):
                self._mojo.step()

        self._env_steps_this_episode += 1

        truncated = False  # Handled by wrappers
        info = self.get_info()
        return (
            self.get_observation(),
            self.calculate_reward(),
            self.should_terminate(),
            truncated,
            info,
        )

    @abstractmethod
    def calculate_reward(self) -> float:
        """Calculate the reward for this step."""
        return 0

    @abstractmethod
    def should_terminate(self) -> bool:
        """Reruns if the environment should terminate this step."""
        return False

    @property
    @abstractmethod
    def task_success(self) -> bool:
        """Determines if the task is successful."""
        pass

    def get_info(self) -> dict:
        """Get info dict."""
        return {}

    def render(
        self,
        camera_name=RENDER_PREVIEW_CAMERA_NAME,
        camera_resolution=[RENDER_PREVIEW_SIZE, RENDER_PREVIEW_SIZE],
        segment=False,
    ):
        """See base."""
        if self._env_steps_this_episode % RENDER_STEP_DIVIDER != 0:
            return None
        if self.preview_renderer is None:
            self.preview_renderer = mujoco.Renderer(
                self._mojo.model,
                camera_resolution[1],
                camera_resolution[0],
            )

        if segment:
            self.preview_renderer.enable_segmentation_rendering()
            options = mujoco.MjvOption()
            options.geomgroup = [1, 1, 1, 0, 0, 0]
            self.preview_renderer.update_scene(self._mojo.data, camera_name, options)
        else:
            self.preview_renderer.disable_segmentation_rendering()
            self.preview_renderer.update_scene(self._mojo.data, camera_name)

        image_to_show = self.preview_renderer.render()
        if self.render_mode == "human":
            # Show mujoco GUI
            self.mujoco_renderer.render("human")
            self._render_human_mode(image_to_show)
            image_to_show = None
        return image_to_show

    def close(self):
        """See base."""
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
        if self.obs_renderer is not None:
            self.obs_renderer.close()
        if self.preview_renderer is not None:
            self.preview_renderer.close()
        if self._render_fig is not None:
            plt.close()
            self._render_fig = self._render_ax = self._render_im = None

    def _render_human_mode(self, image: np.ndarray):
        if self._render_fig is None:
            self._render_fig = plt.figure(
                figsize=(image.shape[1] / 100, image.shape[0] / 100),
            )
            self._render_ax = self._render_fig.add_subplot(111)
            self._render_im = self._render_ax.imshow(image)
            self._render_ax.set_xticks([])  # Remove axis ticks
            self._render_ax.set_yticks([])
            # Remove white space around the plot
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.show(block=False)
        self._render_im.set_array(image)
        plt.pause(0.01)
