"""Robot."""

from typing import List, Optional, Tuple

import numpy as np
from mojo import Mojo
from mojo.elements import Body, Camera, Geom, MujocoElement, Site
from pyquaternion import Quaternion

from dSkill.sim.consts import GRIPPER_SEG_GROUP, ROBOTIQ_MODEL, ROBOTIQ_XML, UR5_XML

ROBOT_START_JOINT_POSITIONS = np.deg2rad([90, -120, -90, 30, 90, 0, 0])
NUM_ARM_JOINTS = 6
NUM_JOINTS_TO_CTRL = 4
ARM_JOINTS_CTRL_IDXS = [1, 2, 3]
GRIPPER_JOINTS_CTRL_IDXS = [6]
NUMBER_OF_RESET_PHYSICS_STEPS = 1000
GRIPPER_OPEN_SETPOINT = 0
GRIPPER_CLOSE_SETPOINT = 255
DELTA_Q = 60

RIGHT_GRIPPER_PAD_NAMES = [
    f"ur5e/robotiq_{ROBOTIQ_MODEL}/right_pad1",
    f"ur5e/robotiq_{ROBOTIQ_MODEL}/right_pad2",
    f"ur5e/robotiq_{ROBOTIQ_MODEL}/right_pad3",
]
LEFT_GRIPPER_PAD_NAMES = [
    f"ur5e/robotiq_{ROBOTIQ_MODEL}/left_pad1",
    f"ur5e/robotiq_{ROBOTIQ_MODEL}/left_pad2",
    f"ur5e/robotiq_{ROBOTIQ_MODEL}/left_pad3",
]

ATTACHMENT_SITE = "attachment_site"
DELTA_Q = 60


class Robot:
    """Robot."""

    def __init__(self, mojo: Mojo):
        """Init."""
        self._mojo = mojo
        self._robot_body = mojo.load_model(UR5_XML)
        self._robot_arm_joints = self._robot_body.joints
        gripper_attachment_site = Site.get(
            mojo,
            "ur5e/attachment_site",
            parent=self._robot_body,
        )
        self._robotiq_body = mojo.load_model(
            ROBOTIQ_XML,
            parent=gripper_attachment_site,
        )
        self._robot_gripper_joints = self._robotiq_body.joints
        self._robot_gripper_seg_geom_ids = [
            geom.id
            for geom in self._robotiq_body.geoms
            if self._mojo.physics.bind(geom.mjcf).group == GRIPPER_SEG_GROUP
        ]

        self._robotiq_tcp = Site.get(mojo, f"ur5e/robotiq_{ROBOTIQ_MODEL}/pinch")
        self._right_gripper_pad_geoms = [
            Geom.get(mojo, pad_name) for pad_name in RIGHT_GRIPPER_PAD_NAMES
        ]
        self._left_gripper_pad_geoms = [
            Geom.get(mojo, pad_name) for pad_name in LEFT_GRIPPER_PAD_NAMES
        ]
        self._bounds = self._get_bounds()
        self.gripper_joints_ctrl_idxs = GRIPPER_JOINTS_CTRL_IDXS

        self._initialize_cameras(gripper_attachment_site)

        # Initialize the Gripper class
        self.gripper = Gripper(
            self._mojo,
            self._robotiq_body,
            self._right_gripper_pad_geoms,
            self._left_gripper_pad_geoms,
        )

    def _initialize_cameras(self, gripper_attachment_site):
        """Initialize cameras for the robot."""
        camera_rotation = Quaternion(axis=[1, 0, 0], angle=np.deg2rad(90))
        self.third_person_camera = Camera.create(
            self._mojo,
            position=[0.8, -1.0, 0.4],
            fovy=45,
            quaternion=camera_rotation.elements,
        )
        self.third_person_camera.mjcf.name = "third_person_camera"

        wrist_camera_parent = Body.create(
            self._mojo,
            parent=gripper_attachment_site.parent,
            position=[0, 0.1, 0],
            quaternion=[-1, 1, 0, 0],
        )

        self._initialize_wrist_cameras(wrist_camera_parent)

    def _initialize_wrist_cameras(self, wrist_camera_parent):
        """Initialize wrist cameras."""
        upper_camera_rotation = Quaternion(axis=[1, 0, 0], angle=np.deg2rad(175))
        self.upper_wrist_camera = Camera.create(
            self._mojo,
            parent=wrist_camera_parent,
            position=[-0.036, -0.081, 0.017],
            fovy=45,
            quaternion=upper_camera_rotation.elements,
        )
        self.upper_wrist_camera.mjcf.name = "upper_wrist_camera"

        lower_camera_rotation = Quaternion(axis=[1, 0, 0], angle=np.deg2rad(185))
        self.lower_wrist_camera = Camera.create(
            self._mojo,
            parent=wrist_camera_parent,
            position=[-0.036, +0.081, 0.017],
            fovy=45,
            quaternion=lower_camera_rotation.elements,
        )
        self.lower_wrist_camera.mjcf.name = "lower_wrist_camera"

    def reset(self):
        self._mojo.data.ctrl[:] = ROBOT_START_JOINT_POSITIONS[:]
        # # need to step simulator for forward IK to be set

        self._mojo.data.qpos[:NUM_ARM_JOINTS] = ROBOT_START_JOINT_POSITIONS[
            :NUM_ARM_JOINTS
        ]
        self._mojo.data.qvel[:NUM_ARM_JOINTS] *= 0
        self._mojo.data.qacc[:NUM_ARM_JOINTS] *= 0

        for _ in range(NUMBER_OF_RESET_PHYSICS_STEPS):
            self._mojo.step()

    def control(self, action: np.ndarray, delta: bool = True):
        """Apply control."""
        ctrl = self._mojo.data.ctrl.copy()
        ctrl[ARM_JOINTS_CTRL_IDXS] += action[:3]
        ctrl[GRIPPER_JOINTS_CTRL_IDXS] = action[-1]

        ctrl = np.clip(
            ctrl,
            self._bounds[:, 0],
            self._bounds[:, 1],
        )
        self._mojo.data.ctrl = ctrl

    @property
    def tcp_pose(self):
        return np.concatenate(
            [
                self._robotiq_tcp.get_position(),
                self._robotiq_tcp.get_quaternion(),
            ],
        )

    def get_scoped_camera_name(self, camera_name):
        camera_attr = getattr(self, camera_name, None)
        namescope = camera_attr.mjcf.namescope.name
        namescope = "" if namescope == "scene" else f"{namescope}/"

        if camera_attr is None:
            raise AttributeError(f"{camera_name} does not exist in Robot")
        return f"{namescope}{camera_attr.mjcf.name}"

    def _get_bounds(self) -> np.ndarray:
        bounds = self._mojo.model.actuator_ctrlrange.copy().astype(np.float32)
        qpos1_deg = np.rad2deg(ROBOT_START_JOINT_POSITIONS[1])
        qpos2_deg = np.rad2deg(ROBOT_START_JOINT_POSITIONS[2])
        qpos3_deg = np.rad2deg(ROBOT_START_JOINT_POSITIONS[3])

        # Setting the bounds with the updated angle values
        bounds[1][0] = np.deg2rad(qpos1_deg - DELTA_Q)
        bounds[1][1] = np.deg2rad(qpos1_deg + DELTA_Q)
        bounds[2][0] = np.deg2rad(qpos2_deg - DELTA_Q)
        bounds[2][1] = np.deg2rad(qpos2_deg + DELTA_Q)
        bounds[3][0] = np.deg2rad(qpos3_deg - DELTA_Q)
        bounds[3][1] = np.deg2rad(qpos3_deg + DELTA_Q)
        return bounds


class Gripper:
    def __init__(
        self,
        mojo: Mojo,
        robotiq_body: Body,
        right_gripper_pad_geoms: List[Geom],
        left_gripper_pad_geoms: List[Geom],
    ):
        self._mojo = mojo
        self._robotiq_body = robotiq_body
        self._right_gripper_pad_geoms = right_gripper_pad_geoms
        self._left_gripper_pad_geoms = left_gripper_pad_geoms

    def has_collided(self, obj: MujocoElement) -> bool:
        """
        Check if any part of the Robotiq body has collided with the given object.
        Args:
            obj (MujocoElement): The object to check for collisions with.
        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        return np.any([geom.has_collided(obj) for geom in self._robotiq_body.geoms])

    def object_grasped(self, obj: MujocoElement) -> bool:
        """
        Check if the Robotiq gripper has grasped the given object.
        This is determined by the presence of object collision normals on both
        left/right gripper pads.
        Args:
            obj (MujocoElement): The object to check for a grasp.
        Returns:
            bool: True if the object is grasped, False otherwise.
        """
        obj_colliders = self._get_collidable_geometries(obj)
        all_right_normals, all_left_normals = self._collect_collision_normals(
            obj_colliders
        )
        return np.any(all_right_normals) and np.any(all_left_normals)

    def opposing_grasp(self, obj: MujocoElement) -> bool:
        """
        Determine if the Robotiq gripper is performing an opposing grasp on the given object.
        An opposing grasp is identified when the normals from the object collision points on
        left/right gripper pads are opposing each other.
        Args:
            obj (MujocoElement): The object to check for an opposing grasp.
        Returns:
            bool: True if an opposing grasp is detected, False otherwise.
        """
        obj_colliders = self._get_collidable_geometries(obj)
        all_right_normals, all_left_normals = self._collect_collision_normals(
            obj_colliders
        )

        for right_normal, left_normal in zip(all_right_normals, all_left_normals):
            if self._contains_nan(right_normal) or self._contains_nan(left_normal):
                continue
            if np.dot(right_normal, left_normal) < 0:
                return True

        return False

    def _get_collidable_geometries(self, obj: MujocoElement) -> List[Geom]:
        return [g for g in obj.geoms if g.is_collidable()]

    def _collect_collision_normals(
        self, obj_colliders: List[Geom]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        all_right_collision_normals, all_left_collision_normals = [], []

        for collider in obj_colliders:
            right_normals, left_normals = self._get_gripper_collision_normals(collider)
            all_right_collision_normals.extend(right_normals)
            all_left_collision_normals.extend(left_normals)

        return all_right_collision_normals, all_left_collision_normals

    def _contains_nan(self, normal: np.ndarray) -> bool:
        return np.any(np.isnan(normal))

    def _get_gripper_collision_normals(
        self, obj: MujocoElement
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        right_normals = [
            self._get_pad_collision_normal(pad, obj)
            for pad in self._right_gripper_pad_geoms
        ]
        left_normals = [
            self._get_pad_collision_normal(pad, obj)
            for pad in self._left_gripper_pad_geoms
        ]

        return [n for n in right_normals if n is not None], [
            n for n in left_normals if n is not None
        ]

    def _get_pad_collision_normal(self, pad: Geom, other: Geom) -> Optional[np.ndarray]:
        pad_geom_id = self._mojo.physics.bind(pad.mjcf).element_id
        other_geom_id = self._mojo.physics.bind(other.mjcf).element_id

        for contact in self._mojo.physics.data.contact:
            if contact.dist > 1e-8:
                continue
            if (contact.geom1 == pad_geom_id and contact.geom2 == other_geom_id) or (
                contact.geom2 == pad_geom_id and contact.geom1 == other_geom_id
            ):
                normal = contact.frame[:3] / np.linalg.norm(contact.frame[:3])
                return normal if contact.geom1 == pad_geom_id else -normal
        return None
