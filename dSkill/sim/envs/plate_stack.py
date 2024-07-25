"""VerticalSlide task."""

import logging

import numpy as np
from gymnasium import spaces
from mojo.elements import Geom, Joint, MujocoElement
from pyquaternion import Quaternion

from dSkill.sim.consts import ASSETS_PATH, ROBOTIQ_MODEL
from dSkill.sim.envs.core import dSkillGymEnv
from dSkill.utils.reward_utils import long_tail_tolerance

NUMBER_OF_RESET_PHYSICS_STEPS = 1000
NUMBER_OF_PLATES = 3
PLATE_START_POS = [0.7, -0.14, 0.02]
ROBOT_START_JOINT_POSITIONS = [1.57, -1.76, -2.04, -0.503, 1.57, 1.57, 0]

# For task success criteria
HORIZONTAL_THRESHOLD = 0.1
HEIGHT_THRESHOLD = 0.1
VELOCITY_THRESHOLD = 1e-2

log = logging.getLogger(__name__)


class PlateStack(dSkillGymEnv):
    """Lift plate from pile."""

    def __init__(self, use_sparse_reward: bool = False, *args, **kwargs):

        self.use_sparse_reward = use_sparse_reward
        super().__init__(*args, **kwargs)

    def initialise_env(self):
        super().initialise_env()

        plate_asset_path = str(ASSETS_PATH / "props" / "plate" / "plate.xml")
        self._plates = [
            self._mojo.load_model(plate_asset_path) for _ in range(NUMBER_OF_PLATES)
        ]  # Adjust copy method based on actual API
        for plate in self._plates:
            plate.set_kinematic(True)

        self.stack_plates()
        self._target_plate_colliders = [
            g for g in self._plates[-1].geoms if g.is_collidable()
        ]

        self._floor = Geom.get(self._mojo, "floor")

    def stack_plates(self):
        for idx, plate in enumerate(self._plates):
            plate_position = PLATE_START_POS.copy()
            plate_position[2] += idx * plate_position[2]
            plate.set_position(plate_position)

            quat = Quaternion(axis=[1, 0, 0], degrees=90)
            plate.set_quaternion(quat.elements)

            bound_plate = self._mojo.physics.bind(plate.mjcf.freejoint)
            bound_plate.qvel *= 0
            bound_plate.qacc *= 0

    def on_reset(self):
        # Now close gripper
        gripper_right_driver_joint = Joint.get(
            self._mojo, f"ur5e/robotiq_{ROBOTIQ_MODEL}/right_driver_joint"
        )
        gripper_left_driver_joint = Joint.get(
            self._mojo, f"ur5e/robotiq_{ROBOTIQ_MODEL}/left_driver_joint"
        )
        self._mojo.physics.bind(gripper_right_driver_joint.mjcf).qpos = 0.6
        self._mojo.physics.bind(gripper_left_driver_joint.mjcf).qpos = 0.6

        JOINTS = 7
        ctrl = self._mojo.data.ctrl.copy()[:JOINTS]
        ctrl[-1] = 0
        self._mojo.data.ctrl[:JOINTS] = ctrl

        self.stack_plates()
        self._mojo.data.ctrl[:] = ROBOT_START_JOINT_POSITIONS[:]

        for _ in range(NUMBER_OF_RESET_PHYSICS_STEPS):
            self._mojo.step()

        self.start_plate_positions = [plate.get_position() for plate in self._plates]

    def _check_plate_stack(self) -> bool:
        return all(
            np.linalg.norm(plate.get_position() - start_position) < HORIZONTAL_THRESHOLD
            for plate, start_position in zip(
                self._plates[:-1], self.start_plate_positions[:-1]
            )
        )

    def _is_static(self, plate: MujocoElement) -> bool:
        bound_plate = self._mojo.physics.bind(plate.mjcf.freejoint)
        return np.all(np.abs(bound_plate.qvel) < VELOCITY_THRESHOLD)

    @property
    def task_success(self) -> bool:
        """
        Determine if the plate stacking task is successful based on several criteria.
        """
        plate_stack_stable = self._check_plate_stack()

        target_plate_grasped = self._robot.gripper.object_grasped(
            self._plates[-1]
        ) and self._robot.gripper.opposing_grasp(self._plates[-1])
        target_plate_z_diff = (
            self._plates[-1].get_position()[2] - self.start_plate_positions[-1][2]
        )
        target_plate_at_correct_height = abs(target_plate_z_diff) > HEIGHT_THRESHOLD
        target_plate_horizontal_movement = np.linalg.norm(
            self._plates[-1].get_position()[:2] - self.start_plate_positions[-1][:2]
        )
        target_plate_minimal_horizontal_displacement = (
            target_plate_horizontal_movement < HORIZONTAL_THRESHOLD
        )
        target_plate_stable = self._is_static(self._plates[-1])

        return (
            plate_stack_stable
            and target_plate_grasped
            and target_plate_at_correct_height
            and target_plate_minimal_horizontal_displacement
            and target_plate_stable
        )

    def calculate_reward(self) -> float:
        """
        Calculate the reward for the plate stacking task.

        Returns:
        float: The calculated reward based on task performance.
        """

        if self.use_sparse_reward:
            reward = 1.0 if self.task_success else 0.0
        else:
            reward = self._calculate_dense_reward()

        self._latest_reward = reward
        return reward

    def _calculate_dense_reward(self) -> float:

        def calculate_dist_reward(dist: float) -> float:
            return long_tail_tolerance(
                dist,
                -0,
                0,
                0.01,
                value_at_margin=0.1,
            )

        # Reward base plates not moving
        plate_positions = zip(self._plates[:-1], self.start_plate_positions[:-1])
        static_plate_reward = 0
        for idx, (plate, start_position) in enumerate(plate_positions):
            reward = 1.0 * long_tail_tolerance(
                np.linalg.norm(plate.get_position() - start_position),
                -0,
                0,
                0.01,
                value_at_margin=0.1,
            )
            static_plate_reward += reward
        static_plate_reward *= 1.0 / (len(self._plates[:-1]))

        # Reward target plate being securely grasped
        gripper_plate_reward = 0
        if self._robot.gripper.object_grasped(self._plates[-1]):
            gripper_plate_reward += 1.0
        if self._robot.gripper.opposing_grasp(self._plates[-1]):
            gripper_plate_reward += 1.0

        # Reward target plate height
        target_plate_z_diff = (
            self._plates[-1].get_position()[2] - self.start_plate_positions[-1][2]
        )
        target_plate_height_reward = long_tail_tolerance(
            target_plate_z_diff,
            -0,
            0,
            0.1,
            value_at_margin=0.1,
        )

        # Penalize horizontal movement of the target plate
        target_plate_horizontal_movement = np.linalg.norm(
            self._plates[-1].get_position()[:2] - self.start_plate_positions[-1][:2]
        )
        target_plate_horizontal_penalty = -0.5 * calculate_dist_reward(
            target_plate_horizontal_movement
        )

        # Reward for keeping the target plate steady
        if target_plate_z_diff > HEIGHT_THRESHOLD:
            target_plate_stable_reward = (
                0.5  # Additional reward for keeping it steady above the pile
            )
        else:
            target_plate_stable_reward = 0

        return (
            static_plate_reward
            + gripper_plate_reward
            + target_plate_height_reward
            + target_plate_stable_reward
            + target_plate_horizontal_penalty
        )

    def get_observation_space(self) -> spaces.Space:
        space = super().get_observation_space()
        internal_dict = dict(space)
        internal_dict.update(
            {
                "plate_in_w": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(7,),
                    dtype=np.float32,
                ),
                "tcp_in_w": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(7,),
                    dtype=np.float32,
                ),
            },
        )
        return spaces.Dict(internal_dict)

    def get_observation(self) -> dict[str, np.ndarray]:
        internal_dict = super().get_observation()

        plate_in_w = np.concatenate(
            (self._plates[-1].get_position(), self._plates[-1].get_quaternion())
        )

        tcp_in_w = self._robot.tcp_pose

        internal_dict.update(
            {
                "plate_in_w": plate_in_w.astype(np.float32),
                "tcp_in_w": tcp_in_w.astype(np.float32),
            },
        )
        return internal_dict

    def should_terminate(self) -> bool:
        should_term = super().should_terminate()
        gripper_hit_floor = [
            pgeom.has_collided(self._floor) for pgeom in self._robot._robotiq_body.geoms
        ]

        return (
            should_term
            or np.any(gripper_hit_floor)
            or (self.use_sparse_reward and self.task_success)
        )


class PlateStackVision(PlateStack):

    def __init__(self, use_priviledged_info, *args, **kwargs):
        self.use_priviledged_info = use_priviledged_info
        super().__init__(*args, **kwargs)
        log.info(f"Use priviledged info: {self.use_priviledged_info}.")

    def get_observation_space(self) -> spaces.Space:
        if self.use_priviledged_info:
            space = super().get_observation_space()
        else:
            space = dSkillGymEnv.get_observation_space(self)

        return space

    def get_observation(self) -> dict[str, np.ndarray]:
        if self.use_priviledged_info:
            internal_dict = super().get_observation()
        else:
            internal_dict = dSkillGymEnv.get_observation(self)
        return internal_dict
