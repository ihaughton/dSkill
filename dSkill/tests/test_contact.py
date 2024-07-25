import unittest

from mojo.elements import Joint
import numpy as np

from dSkill.sim.envs.plate_stack import PlateStack
from dSkill.sim.consts import ROBOTIQ_MODEL

NUMBER_OF_RESET_PHYSICS_STEPS = 1000
NUMBER_ROBOT_CTRL_JOINTS = 7
ROBOT_START_JOINT_POSITIONS_OPPOSITE = [1.57, -1.76, -2.04, 0.69, 1.57, 1.57, 0]
ROBOT_START_JOINT_POSITIONS_ALIGNED = [1.57, -1.88, -2.04, -1.01, 1.57, 0, 0]


class TestContact(unittest.TestCase):

    def setUp(self):
        self.env = PlateStack(
            render_mode="human",
            cameras=["ur5e/upper_wrist_camera", "ur5e/lower_wrist_camera"],
        )
        _, _ = self.env.reset()

    def place_object_in_gripper(self):
        self.env._mojo.data.ctrl[:] = ROBOT_START_JOINT_POSITIONS_OPPOSITE[:]
        for _ in range(NUMBER_OF_RESET_PHYSICS_STEPS):
            self.env._mojo.step()
        
        gripper_right_driver_joint = Joint.get(
            self.env._mojo, f"ur5e/robotiq_{ROBOTIQ_MODEL}/right_driver_joint"
        )
        gripper_left_driver_joint = Joint.get(
            self.env._mojo, f"ur5e/robotiq_{ROBOTIQ_MODEL}/left_driver_joint"
        )
        self.env._mojo.physics.bind(gripper_right_driver_joint.mjcf).qpos = 0.6
        self.env._mojo.physics.bind(gripper_left_driver_joint.mjcf).qpos = 0.6

        ctrl = self.env._mojo.data.ctrl.copy()[:NUMBER_ROBOT_CTRL_JOINTS]
        ctrl[-1] = 255
        self.env._mojo.data.ctrl[:NUMBER_ROBOT_CTRL_JOINTS] = ctrl

        for _ in range(NUMBER_OF_RESET_PHYSICS_STEPS):
            self.env._plates[-1].set_position(self.env._robot.tcp_pose[:3] + [0.05, 0, 0])
            self.env._plates[-1].set_quaternion(self.env._robot.tcp_pose[3:])
            bound_object = self.env._mojo.physics.bind(self.env._plates[-1].mjcf.freejoint)
            bound_object.qvel *= 0
            bound_object.qacc *= 0
            self.env._mojo.step()
            
        for _ in range(NUMBER_OF_RESET_PHYSICS_STEPS):
            self.env._mojo.step()
            
    
    def place_gripper_on_floor(self):
        self.env._mojo.data.ctrl[:] = ROBOT_START_JOINT_POSITIONS_ALIGNED[:]
        for _ in range(NUMBER_OF_RESET_PHYSICS_STEPS):
            self.env._mojo.step()

    def test_opposite_collision_normals(self):
        self.place_object_in_gripper()
        assert self.env._robot.gripper.opposing_grasp(self.env._plates[-1])
            
    def test_aligned_collision_normals(self):
        self.place_gripper_on_floor()
        
        right_collision_normals, left_collision_normals = self.env._robot.gripper._get_gripper_collision_normals(self.env._floor)
        
        average_right_collision_normal = np.mean(right_collision_normals, axis=0)
        average_left_collision_normal = np.mean(left_collision_normals, axis=0)
        
        dot_product = np.dot(average_right_collision_normal, average_left_collision_normal)
        assert dot_product > 0, "The gripper collision normals are not pointing in same direction."

if __name__ == '__main__':
    unittest.main()
