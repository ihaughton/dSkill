import unittest

import numpy as np
from dSkill.sim.envs.vertical_slide import VerticalSlide

NUM_ARM_JOINTS = 6

class TestGripperReflex(unittest.TestCase):
    def setUp(self):
        self.env = VerticalSlide(
            render_mode="rgb_array",
            object_name="vention_bracket",
            object_offset=[0, 0, 0.1],
            use_gripper_joints_in_obs=True,
        )
        ob, _ = self.env.reset()

    def test_gripper_reflex(self):
        """Test the gripper reflex mechanism."""
        
        # Create an action with zero values (0 arm action + gripper reflex)
        action = np.zeros(self.env.action_space.shape)
        obs, _ = self.env.reset()
        
        initial_gripper_joints = obs["proprioception"][NUM_ARM_JOINTS:]
        print(f"initial_gripper_joints: {initial_gripper_joints}")
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        gripper_joints_after_action = obs["proprioception"][NUM_ARM_JOINTS:]
        print(f"gripper_joints_after_action: {gripper_joints_after_action}")
        
        equal = np.allclose(initial_gripper_joints, gripper_joints_after_action, atol=1e-2)
        self.assertTrue(equal, "Gripper reflex should be contained in one environment step.")


if __name__ == "__main__":
    unittest.main()
