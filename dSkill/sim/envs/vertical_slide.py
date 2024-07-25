"""Plate stack task."""

import logging

import mujoco
import numpy as np
from gymnasium import spaces
from mojo.elements import Body, Geom, Joint, Site

from dSkill.sim.consts import (
    ASSETS_PATH,
    GRIPPER_SEG_GROUP,
    ROBOTIQ_MODEL,
    TARGET_SEG_GROUP,
)
from dSkill.sim.envs.core import dSkillGymEnv
from dSkill.utils.camera_utils import transform_point_to_frame
from dSkill.utils.reward_utils import alignment_to_z_axis, long_tail_tolerance

NUMBER_OF_RESET_PHYSICS_STEPS = 1000
OBJECT_DISTANCE_BEFORE_TERM = 0.5
TCP_DISTANCE_BEFORE_TERM = 0.5
RANDOM_OFFSET = 0.015
RANDOM_ANGLE = 10
TCP_TO_M0_TRANSFORM = np.array(
    [
        [0, 1, 0, 0],  # New x is old y
        [1, 0, 0, 0],  # New y is old x
        [0, 0, -1, 0],  # New z is negative old z
        [0, 0, 0, 1],  # Homogeneous coordinate
    ]
)
TARGET_THRESHOLD = 0.05
ALIGNMENT_THRESHOLD = 0.1


log = logging.getLogger(__name__)


class VerticalSlide(dSkillGymEnv):
    """Translate object with arm."""

    def __init__(
        self,
        object_name,
        object_offset,
        use_sparse_reward: bool = False,
        *args,
        **kwargs,
    ):
        self.object_name = object_name
        self.object_offset = object_offset
        self.use_sparse_reward = use_sparse_reward
        self.aruco_marker_names = [
            f"ur5e/robotiq_{ROBOTIQ_MODEL}/aruco_0",
            f"{self.object_name}_with_aruco/aruco_1",
            f"{self.object_name}_with_aruco/aruco_2",
        ]

        super().__init__(*args, **kwargs)

    def initialise_env(self):
        super().initialise_env()
        self._object = self._mojo.load_model(
            str(
                ASSETS_PATH
                / "props"
                / self.object_name
                / f"{self.object_name}_with_aruco.xml"
            )
        )
        object_seg_geom_ids = [
            geom.id
            for geom in self._object.geoms
            if self._mojo.physics.bind(geom.mjcf).group == TARGET_SEG_GROUP
        ]
        self.set_segmentation_group(
            GRIPPER_SEG_GROUP, self._robot._robot_gripper_seg_geom_ids
        )
        self.set_segmentation_group(TARGET_SEG_GROUP, object_seg_geom_ids)

        self._debug_aruco_0_start_body = Body.create(self._mojo)
        self._debug_aruco_1_body = Body.create(
            self._mojo, parent=self._debug_aruco_0_start_body
        )
        self._debug_aruco_2_body = Body.create(
            self._mojo, parent=self._debug_aruco_0_start_body
        )

        self._object.set_kinematic(True)
        self._object_center_site = Site.get(
            self._mojo, f"{self.object_name}_with_aruco/center"
        )
        self._latest_reward = 0
        self._aruco_sites = [
            Site.get(self._mojo, aruco_name) for aruco_name in self.aruco_marker_names
        ]
        self._floor = Geom.get(self._mojo, "floor")

    def on_reset(self):
        tcp_pose = self._robot.tcp_pose
        offset = np.array(self.object_offset)
        new_object_position = tcp_pose[:3] + offset
        new_object_orientation = tcp_pose[3:]
        self._ee_start = self._robot.tcp_pose[:3]

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
        ctrl[-1] = 255
        self._mojo.data.ctrl[:JOINTS] = ctrl

        for _ in range(NUMBER_OF_RESET_PHYSICS_STEPS):
            self._object.set_position(new_object_position)
            self._object.set_quaternion(new_object_orientation)

            # have to set the object velocity and accelerations to zero
            bound_object = self._mojo.physics.bind(self._object.mjcf.freejoint)
            bound_object.qvel *= 0
            bound_object.qacc *= 0

            self._mojo.step()

        self._T_m0_2_w = None
        self.p_m1_in_m0, self.p_m2_in_m0 = None, None
        self._tcp_start_pos = self._robot.tcp_pose[:3].copy()

    @property
    def task_success(self) -> bool:
        ee_position = self._robot.tcp_pose[:3]
        target_position = self._object_center_site.get_position()
        distance_to_target = self._distance_to_target(target_position, ee_position)
        alignment_value = alignment_to_z_axis(
            self._aruco_sites[1].get_position(), self._aruco_sites[2].get_position()
        )

        is_close_enough = distance_to_target < TARGET_THRESHOLD
        is_well_aligned = alignment_value < ALIGNMENT_THRESHOLD
        is_grasped = self._robot.gripper.object_grasped(self._object)

        return is_close_enough and is_well_aligned and is_grasped

    def calculate_reward(self) -> float:
        """Calculate the reward for this step."""
        if self.use_sparse_reward:
            reward = 1.0 if self.task_success else 0.0
        else:
            ee_position = self._robot.tcp_pose[:3]
            target_position = self._object_center_site.get_position()

            alignment_value = alignment_to_z_axis(
                self._aruco_sites[1].get_position(), self._aruco_sites[2].get_position()
            )

            distance_reward = long_tail_tolerance(
                self._distance_to_target(target_position, ee_position),
                lower_bound=0,
                upper_bound=0,
                margin=0.12,
                value_at_margin=0.1,
            )

            start_distance_reward = long_tail_tolerance(
                self._ee_from_start(ee_position),
                lower_bound=-0.1,
                upper_bound=0.1,
                margin=1.0,
                value_at_margin=0.1,
            )

            alignment_reward = long_tail_tolerance(
                alignment_value,
                lower_bound=0,
                upper_bound=0,
                margin=0.5,
                value_at_margin=0.1,
            )

            reward = (
                1.0 * distance_reward
                + 0.1 * start_distance_reward
                + 0.2 * alignment_reward
            )

        self._latest_reward = reward
        return reward

    def _distance_to_target(self, target_pos, ee_pos) -> float:
        if target_pos is None or ee_pos is None:
            return 0.0
        return np.linalg.norm(target_pos - ee_pos)

    def _ee_from_start(self, ee_position):
        ee_from_start = np.linalg.norm(ee_position - self._ee_start)
        return ee_from_start

    def get_observation_space(self) -> spaces.Space:
        space = super().get_observation_space()
        internal_dict = dict(space)
        internal_dict.update(
            {
                "p_m1_in_m0": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "p_m2_in_m0": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
            },
        )
        return spaces.Dict(internal_dict)

    def get_observation(self) -> dict[str, np.ndarray]:
        internal_dict = super().get_observation()

        p_m1_in_m0, p_m2_in_m0 = self._get_marker_points_relative_to_marker0()
        internal_dict.update(
            {
                "p_m1_in_m0": p_m1_in_m0.astype(np.float32),
                "p_m2_in_m0": p_m2_in_m0.astype(np.float32),
            },
        )
        self._object_dist_from_m0 = np.linalg.norm(p_m1_in_m0)
        return internal_dict

    def should_terminate(self) -> bool:
        should_term = super().should_terminate()
        object_hit_floor = [
            pgeom.has_collided(self._floor) for pgeom in self._object.geoms
        ]
        object_far_away = self._object_dist_from_m0 > OBJECT_DISTANCE_BEFORE_TERM
        tcp_far_away = (
            np.linalg.norm(self._robot.tcp_pose[:3] - self._tcp_start_pos)
            > TCP_DISTANCE_BEFORE_TERM
        )

        return (
            should_term
            or np.any(object_hit_floor)
            or object_far_away
            or tcp_far_away
            or (self.use_sparse_reward and self.task_success)
        )

    def _get_marker_transforms_in_world_frame(self):
        T_m2w_transforms = []
        for aruco_site in self._aruco_sites:
            T_m2w_transforms.append(np.eye(4))
            T_m2w_transforms[-1][:3, 3] = aruco_site.get_position()
            T_m2w_transforms[-1][:3, :3] = aruco_site.get_matrix()
        return T_m2w_transforms

    def _get_marker_points_relative_to_marker0(self):
        T_m2w_transforms = self._get_marker_transforms_in_world_frame()

        if self._T_m0_2_w is None:
            self._T_m0_2_w = T_m2w_transforms[0]
        p_m1_in_w = T_m2w_transforms[1][:3, 3]
        p_m2_in_w = T_m2w_transforms[2][:3, 3]

        p_m1_in_m0 = transform_point_to_frame(np.linalg.inv(self._T_m0_2_w), p_m1_in_w)
        p_m2_in_m0 = transform_point_to_frame(np.linalg.inv(self._T_m0_2_w), p_m2_in_w)

        return p_m1_in_m0, p_m2_in_m0


class VerticalSlideDomainRand(VerticalSlide):
    """Extension of VerticalSlide class with position and friction randomization."""

    def __init__(self, friction_lower, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.friction_lower = friction_lower
        log.info(f"Friction lower bound: {self.friction_lower}.")

    def initialise_env(self):
        super().initialise_env()
        for geom in self._object.geoms:
            if geom.mjcf.name == "collider":
                self._object_collider = geom
                break
        else:
            msg = "Geom 'collider' not found for friction randomization."
            raise RuntimeError(msg)

    def apply_rand_friction_to_object(self):
        """Randomizes the object's collider geom friction coefficients."""
        mu_slide = np.random.uniform(self.friction_lower, 1.0)
        mu_torsion, mu_roll = 0.005, 0.0001

        friction = [mu_slide, mu_torsion, mu_roll]
        self._object_collider._mojo.physics.bind(
            self._object_collider.mjcf
        ).friction = friction

    def apply_rand_transform_to_object(self):
        """Applies random translation and rotation to the object."""
        offset = np.random.uniform(-RANDOM_OFFSET, RANDOM_OFFSET, size=(3,))
        offset[1] = 0  # Y offset set to 0

        self._object.set_position(self._object.get_position() + offset)

        theta = np.radians(np.random.uniform(-RANDOM_ANGLE, RANDOM_ANGLE))
        quat_offset = np.array([np.cos(theta / 2), 0, np.sin(theta / 2), 0])
        quat_offset = quat_offset / np.linalg.norm(quat_offset)

        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, quat_offset, self._object.get_quaternion())
        self._object.set_quaternion(quat)

    def on_reset(self):
        """Random offset and rotation on reset, along with friction."""
        super().on_reset()
        self.apply_rand_transform_to_object()
        self.apply_rand_friction_to_object()


class VerticalSlideVision(VerticalSlideDomainRand):

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
            p_m1_in_m0, _ = self._get_marker_points_relative_to_marker0()
            self._object_dist_from_m0 = np.linalg.norm(p_m1_in_m0)
        return internal_dict
