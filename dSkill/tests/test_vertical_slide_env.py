import unittest

import numpy as np

from dSkill.sim.envs.vertical_slide import VerticalSlide
from dSkill.sim.robot import ROBOT_START_JOINT_POSITIONS

OBJECT = "vention_bracket"


class TestReward(unittest.TestCase):
    def setUp(self):
        self.env = VerticalSlide(
            object_name=OBJECT,
            object_offset=[0, 0, 0.1],
        )
        self.env.reset()

    def test_observation(self):
        obs = self.env.get_observation()
        expected_shapes = {
            "gripper_reflex": (1,),
            "p_m1_in_m0": (3,),
            "p_m2_in_m0": (3,),
            "proprioception": (6,),
        }
        # Check if each array in the dictionary matches its expected shape
        for key, expected_shape in expected_shapes.items():
            with self.subTest(key=key, expected_shape=expected_shape):
                array_shape = obs[key].shape
                self.assertEqual(
                    array_shape,
                    expected_shape,
                    f"Shape of '{key}' is {array_shape}, expected {expected_shape}",
                )

    def test_step(self):
        obs, _, terminated, _, _ = self.env.step(
            np.concatenate((np.zeros(3), [1.0]))
        )
        expected_ctrl = ROBOT_START_JOINT_POSITIONS
        expected_ctrl[-1] = 255
        assert not terminated
        assert np.allclose(self.env._mojo.data.ctrl[:7], expected_ctrl, atol=1e-4)
        assert np.allclose(obs["proprioception"][:6], expected_ctrl[:6], atol=0.1)

    def test_reset(self):
        obs, _ = self.env.reset()
        expected_ctrl = ROBOT_START_JOINT_POSITIONS
        expected_ctrl[-1] = 255
        assert np.allclose(self.env._mojo.data.ctrl[:7], expected_ctrl, atol=0.0001)
        assert np.allclose(obs["proprioception"][:6], expected_ctrl[:6], atol=0.1)
        
    def test_state_gripper_reflex(self):
        for i in range(10):
            # Reflex not triggered
            random_action = np.concatenate(
                (np.random.uniform(low=-0.04, high=0.04, size=3),
                [1]),
            )
            obs, _, _, _, _ = self.env.step(random_action)
            assert obs['gripper_reflex'] == [0]
            # Reflex triggered
            random_action = np.concatenate(
                (np.random.uniform(low=-0.04, high=0.04, size=3),
                [0]),
            )
            obs, _, _, _, _ = self.env.step(random_action)
            assert obs['gripper_reflex'] == [1]
            
    def test_max_reward(self):
        self.env = VerticalSlide(
            object_name=OBJECT,
            object_offset=[0, 0, 0],
        )
        self.env.reset()
        _, reward, _, _, _ = self.env.step(
            np.concatenate((np.zeros(3), [1.0]))
        )
        assert np.isclose(reward, 1.3, atol=0.05)

    def test_low_reward(self):
        self.env = VerticalSlide(
            object_name=OBJECT,
            object_offset=[0, 0, 0.1],
        )
        self.env.reset()
        action = np.concatenate((0.1 * np.ones(3), [0.0]))
        for _ in range(10):
            _, reward, _, _, _ = self.env.step(action)
        assert reward <= 0.3


if __name__ == "__main__":
    unittest.main()
