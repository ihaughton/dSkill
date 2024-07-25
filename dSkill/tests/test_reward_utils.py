import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R

from dSkill.utils.reward_utils import alignment_to_z_axis
from dSkill.sim.envs.vertical_slide import VerticalSlide


def add_45_degree_offset(quaternion, axis='z'):
    return (R.from_quat(quaternion) * R.from_euler(axis, 45, degrees=True)).as_quat()

class TestRewardFunctions(unittest.TestCase):
    
    def setUp(self):
        self.env = VerticalSlide(
            render_mode="rgb_array",
            object_name="vention_bracket",
            object_offset=[0, 0, 0.1],
            use_gripper_joints_in_obs=True,
        )
        _, _ = self.env.reset()
        self.env._ee_start = np.array([0, 0, 0])

    def test_distance_to_target(self):
        self.assertAlmostEqual(self.env._distance_to_target(np.array([1, 1, 1]), np.array([1, 1, 1])), 0.0)
        self.assertAlmostEqual(self.env._distance_to_target(np.array([1, 1, 1]), np.array([4, 5, 6])), 7.0710678118654755)
        self.assertAlmostEqual(self.env._distance_to_target(np.array([0, 0, 0]), np.array([0, 0, 0])), 0.0)
        self.assertAlmostEqual(self.env._distance_to_target(None, np.array([1, 1, 1])), 0.0)
        self.assertAlmostEqual(self.env._distance_to_target(np.array([1, 1, 1]), None), 0.0)

    def test_ee_from_start(self):
        self.assertAlmostEqual(self.env._ee_from_start(np.array([0, 0, 0])), 0.0)
        self.assertAlmostEqual(self.env._ee_from_start(np.array([1, 1, 1])), 1.7320508075688772)
        self.assertAlmostEqual(self.env._ee_from_start(np.array([3, 4, 0])), 5.0)
        self.assertAlmostEqual(self.env._ee_from_start(np.array([6, 8, 0])), 10.0)
    
    def test_perfect_alignment(self):
        self.assertAlmostEqual(alignment_to_z_axis([0, 0, 0], [0, 0, 1]), 0)
        self.assertAlmostEqual(alignment_to_z_axis([1, 2, 3], [1, 2, 4]), 0)
        
    def test_perfect_opposite_alignment(self):
        self.assertAlmostEqual(alignment_to_z_axis([0, 0, 0], [0, 0, -1]), 0)
        self.assertAlmostEqual(alignment_to_z_axis([1, 2, 3], [1, 2, 2]), 0)
        
    def test_perpendicular_alignment(self):
        self.assertAlmostEqual(alignment_to_z_axis([0, 0, 0], [1, 0, 0]), 1)
        self.assertAlmostEqual(alignment_to_z_axis([0, 0, 0], [0, 1, 0]), 1)
        
        

class TestSparseRewardFunctions(unittest.TestCase):
    
    def setUp(self):
        self.env = VerticalSlide(
            render_mode="rgb_array",
            object_name="vention_bracket",
            object_offset=[0, 0, 0.1],
            use_gripper_joints_in_obs=True,
            use_sparse_reward=True,
        )
        _, _ = self.env.reset()
        self.env._ee_start = np.array([0, 0, 0])     
        
        
    def test_sparse_reward_perfect_case(self):        
        tcp_pose = self.env._robot.tcp_pose
        offset = [0.,0.,0.]
        new_object_position = tcp_pose[:3] + offset
        new_object_orientation = tcp_pose[3:]
        
        self.env._object.set_position(new_object_position)
        self.env._object.set_quaternion(new_object_orientation)
        
        self.assertEqual(self.env.calculate_reward(), 1.0)

    def test_sparse_reward_poor_alignment(self):
        tcp_pose = self.env._robot.tcp_pose
        offset = [0.,0.,0.]
        new_object_position = tcp_pose[:3] + offset
        new_object_orientation = add_45_degree_offset(tcp_pose[3:], axis='y')
        
        self.env._object.set_position(new_object_position)
        self.env._object.set_quaternion(new_object_orientation)

        self.assertEqual(self.env.calculate_reward(), 0.0)

    def test_sparse_reward_far_from_target_poor_alignment(self):
        tcp_pose = self.env._robot.tcp_pose
        offset = [2.,0.,0.]
        new_object_position = tcp_pose[:3] + offset
        new_object_orientation = add_45_degree_offset(tcp_pose[3:], axis='y')
        
        self.env._object.set_position(new_object_position)
        self.env._object.set_quaternion(new_object_orientation)
        
        self.assertEqual(self.env.calculate_reward(), 0.0)

    def test_sparse_reward_far_from_target_good_alignment(self):
        tcp_pose = self.env._robot.tcp_pose
        offset = [2.,0.,0.]
        new_object_position = tcp_pose[:3] + offset
        new_object_orientation = tcp_pose[3:]
        
        self.env._object.set_position(new_object_position)
        self.env._object.set_quaternion(new_object_orientation)
        
        self.assertEqual(self.env.calculate_reward(), 0.0)


if __name__ == "__main__":
    unittest.main()
