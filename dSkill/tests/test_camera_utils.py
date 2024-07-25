import pytest
import unittest

import numpy as np
from pyquaternion import \
    Quaternion  # Replace with your actual quaternion import=

from dSkill.sim.consts import ROBOTIQ_MODEL
from dSkill.utils.camera_utils import (is_point_in_frustum,
                                         project_point_to_pixel,
                                         transform_point_to_frame)


OBJECT = "vention_bracket"

class TestCameraTransformations(unittest.TestCase):
    def test_transform_point_to_frame(self):
        frame_transform = np.array(
            [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
        )
        point = np.array([1, 1, 1])
        transformed_point = transform_point_to_frame(frame_transform, point)
        expected_point = np.array([2, 3, 4])
        np.testing.assert_array_almost_equal(transformed_point, expected_point)

    def test_project_point_to_pixel(self):
        camera_intrinsics = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
        camera_extrinsics = np.eye(4)
        point_world = np.array([0, 0, 10])
        pixel_coord = project_point_to_pixel(
            camera_intrinsics, camera_extrinsics, point_world
        )
        expected_pixel_coord = np.array([320, 240])
        np.testing.assert_array_almost_equal(pixel_coord, expected_pixel_coord)

    def test_is_point_in_frustum(self):
        fov = np.radians(90)
        aspect_ratio = 16 / 9
        near = 0.1
        point_cam = np.array([0, 0, 0.5])  # Inside frustum
        self.assertTrue(is_point_in_frustum(point_cam, fov, aspect_ratio, near))

        point_cam_outside = np.array([10, 10, 0.5])  # Outside frustum
        self.assertFalse(
            is_point_in_frustum(point_cam_outside, fov, aspect_ratio, near)
        )


import copy

import cv2
import mujoco
from dm_control import mjcf
from transforms3d.quaternions import mat2quat, quat2mat

import dSkill.utils.camera_utils as cam_utils
import dSkill.utils.geometry_utils as geom_utils
from dSkill.sim.envs.vertical_slide import VerticalSlide


def draw_3d_unit_vector_on_2d(
    image,
    start_point_3d,
    unit_vector_3d,
    scale,
    color,
    thickness,
    camera_matrix,
    dist_coefs,
):
    """
    Draws a 3D unit vector from a 3D start point onto a 2D image, taking into account camera intrinsics and distortion.

    Parameters:
    - image: The 2D image on which to draw the vector.
    - start_point_3d: A tuple or array of (x, y, z) coordinates for the 3D start point of the vector in camera coordinates.
    - unit_vector_3d: A tuple or array of (x, y, z) components of the 3D unit vector indicating direction.
    - scale: A scale factor to visually represent the length of the vector.
    - color: The color of the vector, as a tuple in BGR format (e.g., (255, 0, 0) for blue).
    - thickness: The thickness of the vector line.
    - camera_matrix: The camera intrinsic matrix.
    - dist_coefs: The distortion coefficients.
    """
    # Scale the 3D unit vector for visualization
    scaled_vector_3d = np.multiply(unit_vector_3d, scale)

    # Calculate the 3D end point of the vector
    end_point_3d = np.add(start_point_3d, scaled_vector_3d)

    # Project the start and end points onto the 2D image plane, considering distortion
    start_point_2d, _ = cv2.projectPoints(
        start_point_3d.reshape(1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        camera_matrix,
        dist_coefs,
    )
    end_point_2d, _ = cv2.projectPoints(
        end_point_3d.reshape(1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        camera_matrix,
        dist_coefs,
    )

    # Convert to integer pixel coordinates
    start_point_2d = tuple(start_point_2d[0][0].astype(int))
    end_point_2d = tuple(end_point_2d[0][0].astype(int))

    # Draw the line representing the projected vector
    cv2.line(image, start_point_2d, end_point_2d, color, thickness)
    return image


@pytest.mark.skip(reason="All tests in this class are skipped because they are not implemented yet.")
class TestArucoTracking(unittest.TestCase):
    # TODO: Some work needs to be done to make these test work
    def setUp(self):
        self.CAMERA_NAME = "aruco_cam"
        self.OBJECT_NAME = f"{OBJECT}_with_aruco/{OBJECT}"
        self.ARUCO_0_NAME = f"{ROBOTIQ_MODEL}/aruco_0"
        self.ARUCO_1_NAME = f"{OBJECT}/aruco_1"
        self.ARUCO_2_NAME = f"{OBJECT}/aruco_2"
        self.ARUCO_NAME = {
            0: f"{ROBOTIQ_MODEL}/aruco_0",
            1: f"{OBJECT}/aruco_1",
            2: f"{OBJECT}/aruco_2",
        }
        self.ARUCO_LENGTH = 0.05
        self.SAVE_IMG = True
        self.env = VerticalSlide(
            object_name=OBJECT,
            object_offset=[0, 0, 0.1],
        )
        self.env.reset()

    def test_detect_markers(self):
        obs = self.env.get_observation()
        img = obs[f"rgb_{self.CAMERA_NAME}"]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Call the track_aruco function
        result = cam_utils.track_aruco(
            img,
            self.env._get_camera_intrinsics(self.CAMERA_NAME),
            self.env._get_dist_coeffs(self.CAMERA_NAME),
            aruco_length=self.ARUCO_LENGTH,
        )

        # Check if markers were detected
        self.assertIsNotNone(result)
        ids, tvecs, rvecs, corners = result

        self.assertGreater(ids.size, 0)

    def test_detect_marker_and_compare_to_ground_truth(self):
        obs = self.env.get_observation()
        img = obs[f"rgb_{self.CAMERA_NAME}"]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Detect ArUco markers in the image
        result = cam_utils.track_aruco(
            img,
            self.env._get_camera_intrinsics(self.CAMERA_NAME),
            self.env._get_dist_coeffs(self.CAMERA_NAME),
            aruco_length=self.ARUCO_LENGTH,
        )

        self.assertIsNotNone(result)
        ids, tvecs, rvecs, corners = result

        # Find marker 0 among detected markers
        marker_0_index = np.where(ids == 0)[0]
        self.assertGreaterEqual(len(marker_0_index), 1, "Marker 0 not detected.")
        detected_pos = tvecs[marker_0_index]
        detected_rvec = rvecs[marker_0_index]

        # Transform from camera frame to world frame
        cam_2_world = self.env._get_camera_extrinsics(self.CAMERA_NAME)
        # world_2_cam = np.linalg.inv(cam_2_world)
        cam_2_world = np.dot(cam_2_world, cam_utils.MUJOCO_TO_CV_CAMERA_TRANSFORM)

        detected_2_cam = geom_utils.create_affine_transformation(
            detected_pos, detected_rvec
        )
        detected_2_world = np.dot(cam_2_world, detected_2_cam)
        detected_pos_world = detected_2_world[:3, 3]
        detected_rvec_world, _ = cv2.Rodrigues(detected_2_world[:3, :3])
        detected_rvec_world = np.squeeze(detected_rvec_world)

        # Retrieve the ground truth position and orientation of marker 0
        mujoco_idx = self.env.model.site(self.ARUCO_0_NAME).id
        ground_truth_pos = self.env._mojo.data.site_xpos[mujoco_idx]
        ground_truth_quat = mat2quat(self.env._mojo.data.site_xmat[mujoco_idx])
        ground_truth_rvec = geom_utils.quaternion_to_rvec(ground_truth_quat)

        np.testing.assert_almost_equal(detected_pos_world, ground_truth_pos, decimal=2)

        detected_2_world_quat = geom_utils.rvec_to_quaternion(detected_rvec_world)
        q_diff, ang_diff = geom_utils.calculate_angular_difference(
            detected_2_world_quat, ground_truth_quat
        )
        np.testing.assert_almost_equal(ang_diff, 0.0, decimal=1)

    def test_ground_truth_compared_to_detected_marker(self):
        for i in range(10):
            object_transform = np.eye(4)
            object_transform[:3, 3] = self.object_pos
            object_transform[:3, :3] = quat2mat([0, 0, 0, 1])

            offset_transform = np.eye(4)
            offset_transform[:3, 3] = np.random.uniform(-0.05, 0.05, size=3)
            offset_transform[:3, :3] = geom_utils.generate_random_rotation_matrix(
                (-10, -180, -10), (10, 180, 10)
            )
            # offset_transform[:3, 3] = np.array([0,0,0]) #np.random.uniform(-0.0, 0.0, size=3)
            # offset_transform[:3, :3] = geom_utils.generate_random_rotation_matrix((-0, -0, -0), (0, 0, 0))

            object_transform = np.dot(object_transform, offset_transform)
            self.env.set_body_pose(
                self.OBJECT_NAME,
                position=object_transform[:3, 3],
                quaternion=mat2quat(object_transform[:3, :3]),
            )
            mujoco.mj_step(self.env.model, self.env.data, nstep=1)

            obs = self.env.get_observation()
            img = obs[f"rgb_{self.CAMERA_NAME}"]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Detect ArUco markers in the image
            result = cam_utils.track_aruco(
                img,
                self.env._get_camera_intrinsics(self.CAMERA_NAME),
                self.env._get_dist_coeffs(self.CAMERA_NAME),
                aruco_length=self.ARUCO_LENGTH,
            )

            # self.assertIsNotNone(result)
            ids, tvecs, rvecs, corners = result

            # Find marker among detected markers
            # print(f"ids: {ids}")
            # print(f"ids.flatten(): {ids.flatten()}")

            for marker_id in ids.flatten():
                marker_index = np.where(ids == marker_id)
                self.assertGreaterEqual(len(marker_index), 1, "Marker not detected.")
                detected_pos = np.squeeze(tvecs[marker_index])
                detected_rvec = np.squeeze(rvecs[marker_index])

                # Ground truth position and default orientation for ArUco marker 0 in the world frame
                mujoco_idx = self.env.model.site(self.ARUCO_NAME[marker_id]).id
                ground_truth_pos = self.env._mojo.data.site_xpos[mujoco_idx]
                ground_truth_quat = mat2quat(self.env._mojo.data.site_xmat[mujoco_idx])
                ground_truth_rvec, _ = cv2.Rodrigues(
                    quat2mat(ground_truth_quat)
                )  # Convert quaternion to rotation vector

                # Transform the ground truth position and orientation from the world frame to the camera frame
                cam_2_world = self.env._get_camera_extrinsics(self.CAMERA_NAME)
                world_2_cam = np.linalg.inv(cam_2_world)
                world_2_cam = np.dot(
                    cam_utils.MUJOCO_TO_CV_CAMERA_TRANSFORM, world_2_cam
                )

                ground_truth_2_world = geom_utils.create_affine_transformation(
                    ground_truth_pos, ground_truth_rvec
                )
                ground_truth_2_cam = np.dot(world_2_cam, ground_truth_2_world)
                ground_truth_pos_cam = ground_truth_2_cam[:3, 3]
                ground_truth_rvec_cam, _ = cv2.Rodrigues(ground_truth_2_cam[:3, :3])
                ground_truth_rvec_cam = np.squeeze(ground_truth_rvec_cam)

                # print(f"detected_pos: {detected_pos}")
                # print(f"detected_rvec: {detected_rvec}")
                # img_gt = copy.deepcopy(img)
                # Draw the ground truth position and orientation in the camera frame on the image
                cv2.drawFrameAxes(
                    img,
                    self.env._get_camera_intrinsics(self.CAMERA_NAME),
                    self.env._get_dist_coeffs(self.CAMERA_NAME),
                    ground_truth_rvec_cam,
                    ground_truth_pos_cam,
                    self.ARUCO_LENGTH,
                )
                cv2.drawFrameAxes(
                    img,
                    self.env._get_camera_intrinsics(self.CAMERA_NAME),
                    self.env._get_dist_coeffs(self.CAMERA_NAME),
                    detected_rvec,
                    detected_pos,
                    self.ARUCO_LENGTH,
                )

                # Optionally save the image to file
                # if self.SAVE_IMG:
                #     cv2.imwrite(f"test_{i}.png", img)
                #     # cv2.imwrite(f"test_{i}_gt.png", img_gt)

                np.testing.assert_almost_equal(
                    detected_pos, ground_truth_pos_cam, decimal=1
                )

                # detected_quat = geom_utils.rvec_to_quaternion(detected_rvec)
                # ground_truth_quat_cam = geom_utils.rvec_to_quaternion(ground_truth_rvec_cam)
                # # q_diff, ang_diff = geom_utils.calculate_angular_difference(detected_quat, ground_truth_quat_cam)
                # # q_diff, ang_diff = geom_utils.calculate_angular_difference_axis(detected_quat, ground_truth_quat_cam, axis=2)
                # ang_diff = geom_utils.calculate_angular_difference_along_axis(detected_quat, ground_truth_quat_cam, axis=1)

                # print(f"ang_diff_axis_0: {geom_utils.calculate_angular_difference_along_axis(detected_quat, ground_truth_quat_cam, axis=0)}")
                # print(f"ang_diff_axis_1: {geom_utils.calculate_angular_difference_along_axis(detected_quat, ground_truth_quat_cam, axis=1)}")
                # print(f"ang_diff_axis_2: {geom_utils.calculate_angular_difference_along_axis(detected_quat, ground_truth_quat_cam, axis=2)}")

                # # np.testing.assert_almost_equal(ang_diff, 0.0, decimal=1)
                # print(f"ang_diff: {ang_diff}")
                # assert ang_diff < np.radians(30)

    def test_object_orientation_to_ground_truth(self):
        for i in range(100):
            object_transform = np.eye(4)
            object_transform[:3, 3] = self.object_pos
            object_transform[:3, :3] = quat2mat([0, 0, 0, 1])

            offset_transform = np.eye(4)
            offset_transform[:3, 3] = np.random.uniform(-0.02, 0.02, size=3)
            offset_transform[:3, :3] = geom_utils.generate_random_rotation_matrix(
                (-5, -40, -5), (5, 40, 5)
            )
            # offset_transform[:3, 3] = np.array([0,0,0]) #np.random.uniform(-0.0, 0.0, size=3)
            # offset_transform[:3, :3] = geom_utils.generate_random_rotation_matrix((-0, -0, -0), (0, 0, 0))

            object_transform = np.dot(object_transform, offset_transform)
            self.env.set_body_pose(
                self.OBJECT_NAME,
                position=object_transform[:3, 3],
                quaternion=mat2quat(object_transform[:3, :3]),
            )
            mujoco.mj_step(self.env.model, self.env.data, nstep=1)

            obs = self.env.get_observation()
            img = obs[f"rgb_{self.CAMERA_NAME}"]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Detect ArUco markers in the image
            result = cam_utils.track_aruco(
                img,
                self.env._get_camera_intrinsics(self.CAMERA_NAME),
                self.env._get_dist_coeffs(self.CAMERA_NAME),
                aruco_length=self.ARUCO_LENGTH,
            )

            # self.assertIsNotNone(result)
            ids, tvecs, rvecs, corners = result

            # Find marker among detected markers
            # print(f"ids: {ids}")
            # print(f"ids.flatten(): {ids.flatten()}")

            if 1 in ids.flatten() and 2 in ids.flatten():
                # Get object orientation vector
                marker_1_index = np.where(ids == 1)
                marker_1_pos = np.squeeze(tvecs[marker_1_index])
                marker_2_index = np.where(ids == 2)
                marker_2_pos = np.squeeze(tvecs[marker_2_index])
                object_vec = marker_1_pos - marker_2_pos
                object_vec = object_vec / np.linalg.norm(object_vec)

                img = draw_3d_unit_vector_on_2d(
                    img,
                    marker_2_pos,
                    object_vec,
                    1,
                    (0, 0, 255),
                    4,
                    self.env._get_camera_intrinsics(self.CAMERA_NAME),
                    self.env._get_dist_coeffs(self.CAMERA_NAME),
                )

                # Get ground truth orientation from y axis of aruco marker
                mujoco_idx = self.env.model.site(self.ARUCO_NAME[1]).id
                ground_truth_pos = self.env._mojo.data.site_xpos[mujoco_idx]
                ground_truth_quat = mat2quat(self.env._mojo.data.site_xmat[mujoco_idx])
                ground_truth_rvec, _ = cv2.Rodrigues(
                    quat2mat(ground_truth_quat)
                )  # Convert quaternion to rotation vector

                # Transform the ground truth position and orientation from the world frame to the camera frame
                cam_2_world = self.env._get_camera_extrinsics(self.CAMERA_NAME)
                world_2_cam = np.linalg.inv(cam_2_world)
                world_2_cam = np.dot(
                    cam_utils.MUJOCO_TO_CV_CAMERA_TRANSFORM, world_2_cam
                )

                ground_truth_2_world = geom_utils.create_affine_transformation(
                    ground_truth_pos, ground_truth_rvec
                )
                ground_truth_2_cam = np.dot(world_2_cam, ground_truth_2_world)
                ground_truth_pos_cam = ground_truth_2_cam[:3, 3]
                unit_vec = np.array([0.0, 1.0, 0.0])
                ground_truth_vec = np.dot(ground_truth_2_cam[:3, :3], unit_vec)

                img = draw_3d_unit_vector_on_2d(
                    img,
                    marker_2_pos,
                    ground_truth_vec,
                    1,
                    (0, 255, 0),
                    2,
                    self.env._get_camera_intrinsics(self.CAMERA_NAME),
                    self.env._get_dist_coeffs(self.CAMERA_NAME),
                )

                # Optionally save the image to file
                # if self.SAVE_IMG:
                #     cv2.imwrite(f"test_{i}.png", img)
                # cv2.imwrite(f"test_{i}_gt.png", img_gt)

                # lines should be parrallel
                cross_product = np.cross(object_vec[:2], ground_truth_vec[:2])
                print(f"cross_product: {np.linalg.norm(cross_product)}")
                assert np.linalg.norm(cross_product) < 0.1

                # angle_diff = geom_utils.calculate_angular_difference_vec(object_vec, ground_truth_vec)
                # assert angle_diff < np.radians(10)


if __name__ == "__main__":
    unittest.main()
