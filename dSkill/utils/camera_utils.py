"""Camera utils."""

import copy

import cv2
import numpy as np
from pyquaternion import Quaternion

ARUCO_TO_POINT_TRANS = {
    "0": np.array([0.0, 0.0, 0.0]),
    "1": np.array([0.0, -0.1, 0.0]),
    "2": np.array([0.0, +0.1, 0.0]),
}

MUJOCO_TO_CV_CAMERA_TRANSFORM = np.array(
    [
        [1, 0, 0, 0],  # Keeps x coordinate unchanged
        [0, -1, 0, 0],  # Flips y coordinate
        [0, 0, -1, 0],  # Flips z coordinate
        [0, 0, 0, 1],  # Homogeneous coordinate unchanged for no translation
    ],
)


def transform_to_world_frame(detected_pos, detected_rvec, camera_extrinsic_matrix):
    """Transforms the detected marker position and orientation from the
    camera frame to the world frame.

    Parameters:
    - detected_pos (numpy.ndarray): The 3D position of the detected marker
    in the camera frame.
    - detected_rvec (numpy.ndarray): The rotation vector of the
    detected marker in the camera frame.
    - camera_extrinsic_matrix (numpy.ndarray): The 4x4 extrinsic matrix that
    transforms points from the world frame to the camera frame.

    Returns:
    - tuple: Contains two elements:
        - pos_world (numpy.ndarray): The 3D position of the detected marker
        in the world frame.
        - rvec_world (numpy.ndarray): The rotation vector of the detected
        marker in the world frame.
    """
    # Convert rotation vector to rotation matrix
    detected_rmat, _ = cv2.Rodrigues(detected_rvec)

    # Invert the camera extrinsic matrix to get the camera-to-world transformation
    camera_to_world_matrix = np.linalg.inv(camera_extrinsic_matrix)

    # Extract rotation and translation components from the camera-to-world matrix
    R_world = camera_to_world_matrix[:3, :3]
    t_world = camera_to_world_matrix[:3, 3]

    # Ensure detected_pos is a 1D array for dot product
    detected_pos_1d = np.squeeze(detected_pos)

    # Transform the detected position to the world frame
    pos_world = R_world.dot(detected_pos_1d) + t_world

    # Transform the detected orientation to the world frame
    rmat_world = R_world.dot(detected_rmat)

    # Convert the rotation matrix back to a rotation vector
    rvec_world, _ = cv2.Rodrigues(rmat_world)

    return pos_world, rvec_world.flatten()


def get_marker_pixel_position(corners, marker_index):
    """Calculate the pixel position of the center of an ArUco marker given
    its corner points.

    Parameters:
    - corners (list): List of detected corners for each ArUco marker. Each
    element in the list
                      is an array of four corner points (x, y) of the
                      detected marker.
    - marker_index (int): Index of the marker in the 'corners' list for
    which to calculate the pixel position.

    Returns:
    - numpy.ndarray: The pixel coordinates (x, y) of the center of the
    specified marker.
    """
    marker_corners = corners[marker_index][0]
    center_position = np.mean(marker_corners, axis=0)

    return center_position.astype(np.int32)


def transform_point_to_frame(frame_transform, point):
    """Transform a point from world coordinates to camera coordinates.

    This function applies the provided transformation matrix to a point
    in world coordinates,
    converting it to camera coordinates. It assumes the transformation
    matrix is given in
    the camera frame (i.e., the inverse camera transformation matrix).

    Parameters:
    frame_transform (np.ndarray): The 4x4 transformation matrix from
    world to camera coordinates.
    point (np.ndarray): The 3D point in world coordinates to be
    transformed.

    Returns:
    np.ndarray: The transformed 3D point in camera coordinates.
    """
    if not isinstance(frame_transform, np.ndarray) or frame_transform.shape != (4, 4):
        raise ValueError("frame_transform must be a 4x4 numpy array.")
    if not isinstance(point, np.ndarray) or point.shape != (3,):
        raise ValueError("point must be a 3-element numpy array.")

    # Convert the 3D point to homogeneous coordinates by appending a 1
    point_homogeneous = np.append(point, 1)

    # Apply the transformation matrix to the point in homogeneous coordinates
    point_camera_homogeneous = np.dot(frame_transform, point_homogeneous)

    # Convert back to Cartesian coordinates by normalizing with respect to
    # the homogeneous coordinate
    point_camera = point_camera_homogeneous[:3] / point_camera_homogeneous[3]

    return point_camera


def project_point_to_pixel(camera_intrinsics, camera_extrinsics, point_world):
    """Project a 3D point in world coordinates to 2D pixel coordinates.

    This function takes a 3D point in world coordinates, transforms
    it to camera coordinates
    using the camera extrinsic parameters, and then projects it onto
    the image plane using
    the camera intrinsic parameters.

    Parameters:
    - camera_intrinsics (numpy.ndarray): The 3x3 intrinsic camera matrix.
    - camera_extrinsics (numpy.ndarray): The 4x4 extrinsic camera matrix,
    representing the camera pose in the world frame.
    - point_world (numpy.ndarray): The 3D point in world coordinates to
    be projected.

    Returns:
    - numpy.ndarray: The 2D pixel coordinates of the projected point
    in the image.
    """
    # Convert the world point to homogeneous coordinates for transformation
    point_world_homogeneous = np.append(point_world, 1)

    # Transform the point from world to camera coordinates using the camera's
    # extrinsic matrix
    point_camera_homogeneous = camera_extrinsics.dot(point_world_homogeneous)

    # Project the 3D camera coordinates onto the image plane using the
    # intrinsic matrix.
    # Note: We use only the x, y, z components (ignoring the homogeneous component)
    # for this projection.
    point_image_homogeneous = camera_intrinsics.dot(point_camera_homogeneous[:3])

    # Convert from homogeneous image coordinates to 2D pixel coordinates by
    # dividing by the z component
    pixel_coord = point_image_homogeneous[:2] / point_image_homogeneous[2]

    return pixel_coord


def is_point_in_frustum(point_cam, fov, aspect_ratio, near):
    """Determine if a point in camera coordinates is inside the camera's view frustum.

    The view frustum is defined by the field of view, aspect ratio, and
    the near clipping plane.
    This function checks if the given point falls within the frustum
    boundaries established by these parameters.

    Parameters:
    - point_cam (numpy.ndarray): The 3D point in camera coordinates
    to be checked.
    - fov (float): The camera's vertical field of view in radians.
    - aspect_ratio (float): The width-to-height ratio of the camera's
    view.
    - near (float): The distance from the camera to the near clipping
    plane.

    Returns:
    - bool: True if the point is within the frustum, False otherwise.
    """
    # Calculate the height and width of the near clipping plane
    near_height = 2.0 * np.tan(fov / 2) * near
    near_width = near_height * aspect_ratio

    # Unpack the point's coordinates for clarity
    x, y, z = point_cam

    # Check if the point is within the frustum bounds:
    # - x is within the left and right bounds of the near clipping plane
    # - y is within the top and bottom bounds of the near clipping plane
    # - z is beyond the near clipping plane (in front of the camera)
    is_within_x_bounds = -near_width / 2 < x < near_width / 2
    is_within_y_bounds = -near_height / 2 < y < near_height / 2
    is_in_front_of_camera = z > near

    return is_within_x_bounds and is_within_y_bounds and is_in_front_of_camera


def track_aruco(
    img,
    intrinsics,
    dist_coeffs,
    aruco_length=0.05,
):
    """Detects ArUco markers in the given image and estimates their poses.

    Parameters:
    - img (numpy.ndarray): The image in which to detect ArUco markers.
    - intrinsics (numpy.ndarray): The camera intrinsic parameters.
    - dist_coeffs (numpy.ndarray): The camera distortion coefficients.
    - aruco_length (float): The length of the ArUco marker's side in meters.

    Returns:
    - tuple: A tuple containing the ids, translation vectors, and rotation
    vectors of detected markers. Returns None if no markers are detected.
    """
    # Define ArUco dictionary and detector parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()  # _create()

    # Detect ArUco markers
    corners, ids, rejected = cv2.aruco.detectMarkers(
        img,
        aruco_dict,
        parameters=parameters,
    )

    if ids is not None:
        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            aruco_length,
            intrinsics,
            dist_coeffs,
        )

        return ids, tvecs, rvecs, corners

    raise RuntimeError("No ArUco markers detected in the image.")


def camera_to_aruco_poses(
    aruco_params,
    image=None,
    intrinsics=None,
    dist_coeffs=None,
    n_step=0,
):
    target_pos = None
    ee_pos = None

    target_q = None
    ee_q = None

    (ids, tvecs, rvecs) = aruco_params
    for id, rvec, tvec in zip(ids, rvecs, tvecs, strict=False):
        if str(id) not in ARUCO_TO_POINT_TRANS:
            continue
        T_mp = np.eye(4)  # Transform aruco to point
        T_mp[:3, 3] = ARUCO_TO_POINT_TRANS[str(id)]

        T_cm = np.eye(4)  # Transform camera to aruco
        T_cm[:3, 3] = tvec
        R, _ = cv2.Rodrigues(rvec)
        q = Quaternion(matrix=R)
        T_cm[:3, :3] = q.rotation_matrix

        T_cp = np.dot(T_cm, T_mp)  # Transform camera to point

        if id == 0:
            ee_pos = copy.deepcopy(T_cp[:3, 3])
            ee_q = np.array([q.x, q.y, q.z, q.w])

        if id == 1 or id == 2:
            if target_pos is None:
                target_pos = copy.deepcopy(T_cp[:3, 3])
            else:
                target_pos += T_cp[:3, 3]
                target_pos /= 2.0

            target_q = np.array([q.x, q.y, q.z, q.w])

    return target_pos, target_q, ee_pos, ee_q
