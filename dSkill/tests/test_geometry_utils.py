import unittest

import cv2
import numpy as np
from pyquaternion import \
    Quaternion  # Replace with your actual quaternion import
from transforms3d.quaternions import quat2mat

from dSkill.utils.geometry_utils import (calculate_angular_difference,
                                           make_quaternion_positive,
                                           quaternion_to_rvec)


class TestQuaternions(unittest.TestCase):
    def test_quaternion_to_rvec(self):
        angles = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
        axis = np.array([0, 1, 0])  # Using numpy array for the axis

        for angle in angles:
            with self.subTest(angle=angle):
                # Creating the quaternion [w, x, y, z] from angle and axis
                c = np.cos(angle / 2)
                s = np.sin(angle / 2)
                quaternion = [c, s * axis[0], s * axis[1], s * axis[2]]
                rvec = quaternion_to_rvec(quaternion)
                # Convert rvec back to rotation matrix to compare
                rot_matrix, _ = cv2.Rodrigues(rvec)
                # Convert quaternion to rotation matrix for comparison
                expected_rot_matrix = quat2mat(quaternion)
                np.testing.assert_array_almost_equal(
                    rot_matrix,
                    expected_rot_matrix,
                    decimal=6,
                    err_msg=f"Failed for angle: {angle}",
                )

    def test_make_quaternion_positive(self):
        # Test with a quaternion that has a negative scalar part
        q_neg = Quaternion(-1, 0, 0, 0)
        q_pos = make_quaternion_positive(q_neg)
        self.assertTrue(q_pos[0] >= 0, "Scalar component should be non-negative")

        # Test with a quaternion that already has a non-negative scalar part
        q_already_pos = Quaternion(1, 0, 0, 0)
        q_still_pos = make_quaternion_positive(q_already_pos)
        self.assertTrue(
            q_still_pos[0] >= 0, "Scalar component should remain non-negative"
        )

    def test_identical_quaternions(self):
        q1 = Quaternion(axis=[0, 1, 0], angle=0)
        q2 = Quaternion(axis=[0, 1, 0], angle=0)
        _, angle_diff = calculate_angular_difference(q1, q2)
        self.assertAlmostEqual(
            angle_diff,
            0,
            msg="Angular difference should be zero for identical quaternions",
        )

    def test_inverse_quaternion_angles(self):
        # Array of angles to test
        angles = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
        for angle in angles:
            with self.subTest(angle=angle):
                q1 = Quaternion(axis=[0, 1, 0], angle=angle)
                q2 = q1.inverse
                _, angle_diff = calculate_angular_difference(q1, q2)
                # For inverse quaternions, the expected angle difference should be 0 for 2pi (full rotation) and pi for others
                expected_angle = 0 if angle % (2 * np.pi) == 0 else np.pi
                self.assertAlmostEqual(
                    angle_diff,
                    expected_angle,
                    msg=f"Angular difference should be {expected_angle} for inverse quaternion with angle {angle}",
                )

    def test_known_rotation_angles(self):
        # Array of angles to test
        angles = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
        for angle in angles:
            with self.subTest(angle=angle):
                q1 = Quaternion(axis=[0, 1, 0], angle=0)  # Reference quaternion
                q2 = Quaternion(
                    axis=[0, 1, 0], angle=angle
                )  # Quaternion rotated by 'angle'
                _, angle_diff = calculate_angular_difference(q1, q2)
                # Adjust expected angle for the shortest path in quaternion space
                expected_angle = angle if angle <= np.pi else 2 * np.pi - angle
                self.assertAlmostEqual(
                    angle_diff,
                    expected_angle,
                    delta=1e-6,
                    msg=f"Angular difference should be {expected_angle} for a rotation of {angle} radians",
                )


if __name__ == "__main__":
    unittest.main()
