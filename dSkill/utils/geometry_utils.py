"""Geometry utils."""

import copy

import cv2
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from transforms3d.quaternions import mat2quat, quat2mat


def find_relative_transform(T1, T2):
    """Compute the relative affine transformation matrix from T1 to T2.

    Parameters:
    - T1: The first affine transformation matrix in homogeneous coordinates.
    - T2: The second affine transformation matrix in homogeneous coordinates.

    Returns:
    - The relative affine transformation matrix from T1 to T2.
    """
    # Calculate the inverse of T1
    T1_inv = np.linalg.inv(T1)

    # Compute the relative transformation from T1 to T2
    T_rel = np.dot(T2, T1_inv)

    return T_rel


def rotation_matrix_from_angles(roll, pitch, yaw):
    """Create a rotation matrix from roll, pitch, and yaw angles.

    Parameters:
    - roll: The roll angle in degrees.
    - pitch: The pitch angle in degrees.
    - yaw: The yaw angle in degrees.

    Returns:
    - A 3x3 numpy array representing the rotation matrix.
    """
    # Convert the angles from degrees to radians
    angles_rad = np.radians([roll, pitch, yaw])

    # Convert the Euler angles to a rotation matrix
    # The order 'xyz' specifies that the rotations are first around the x-axis (roll),
    # then around the y-axis (pitch), and finally around the z-axis (yaw).
    rotation = Rotation.from_euler("xyz", angles_rad)
    rotation_matrix = rotation.as_matrix()

    return rotation_matrix


def generate_random_rotation_matrix(min_angles, max_angles):
    """Generate a random rotation matrix within specified Euler angle limits.

    Parameters:
    - min_angles: tuple of 3 elements (min_roll, min_pitch, min_yaw) in degrees
    - max_angles: tuple of 3 elements (max_roll, max_pitch, max_yaw) in degrees

    Returns:
    - A 3x3 numpy array representing the rotation matrix.
    """
    # Generate random Euler angles (in radians) within the specified limits
    random_angles_deg = [
        np.random.uniform(low, high)
        for low, high in zip(min_angles, max_angles, strict=False)
    ]
    random_angles_rad = np.radians(random_angles_deg)

    # Convert Euler angles to a rotation matrix
    rotation = Rotation.from_euler("xyz", random_angles_rad)
    rotation_matrix = rotation.as_matrix()

    return rotation_matrix


def quaternion_to_rvec(quaternion):
    """Convert a quaternion to a rotation vector (rvec) using transforms3d and OpenCV.

    Parameters:
    - quaternion (array-like): The quaternion in the format [w, x, y, z].

    Returns:
    - numpy.ndarray: The rotation vector (rvec) representing the same rotation.
    """
    rotation_matrix = quat2mat(quaternion)
    rvec, _ = cv2.Rodrigues(rotation_matrix)

    return rvec.flatten()


def rvec_to_quaternion(rvec):
    """Convert a rotation vector (rvec) to a quaternion.

    Parameters:
    - rvec (numpy.ndarray): The rotation vector representing a 3D rotation.

    Returns:
    - numpy.ndarray: The quaternion [w, x, y, z] representing the same rotation.
    """
    # Convert rotation vector to rotation matrix using Rodrigues' rotation formula
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Convert rotation matrix to quaternion
    quaternion = mat2quat(rotation_matrix)
    # quaternion = quaternion/np.linalg.norm(quaternion)

    return quaternion


def make_quaternion_positive(q):
    """Ensure the scalar component of the quaternion is non-negative.

    If the scalar component of the input quaternion is negative, this function
    inverts all components of the quaternion to ensure the scalar component is positive.
    This is often done to avoid negative zero components and ensure consistency
    in quaternion representations.

    Parameters:
    q (Quaternion): The input quaternion to be adjusted if necessary.

    Returns:
    Quaternion: The adjusted quaternion with a non-negative scalar component.
    """
    if q[0] < 0:
        q = Quaternion(-q[0], -q[1], -q[2], -q[3])  # w,x,y,z
    else:
        q = Quaternion(q[0], q[1], q[2], q[3])  # w,x,y,z
    return q


def create_affine_transformation(position, rotation_vector):
    """Creates a 4x4 affine transformation matrix from a position and rotation vector.

    Args:
    - position (np.ndarray): A 3-element array representing the translation (x, y, z).
    - rotation_vector (np.ndarray): A 3-element array representing the rotation vector.

    Returns:
    - np.ndarray: A 4x4 affine transformation matrix.
    """
    # Convert the rotation vector to a rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Initialize a 4x4 affine transformation matrix with zeros
    affine_matrix = np.zeros((4, 4))

    # Set the top-left 3x3 part to the rotation matrix
    affine_matrix[:3, :3] = rotation_matrix

    # Set the first three elements of the last column to the translation vector
    affine_matrix[:3, 3] = copy.deepcopy(position)

    # Set the last row to make it a proper affine transformation matrix
    affine_matrix[3] = [0, 0, 0, 1]

    return affine_matrix


def calculate_angular_difference_vec(vec1, vec2):
    """Calculate the angular difference between two vectors.

    This function computes the angle between two vectors in a Euclidean space.
    The angular
    difference is derived from the dot product and the magnitudes of the
    vectors, giving
    the cosine of the angle between them. The result is the angle in radians
    between the two vectors.

    Parameters:
    vec1 (array_like): The first vector in the Euclidean space.
    vec2 (array_like): The second vector in the Euclidean space.

    Returns:
    float: The angular difference between the two vectors in radians.
    """
    # Calculate the cosine of the angle between the two vectors
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0,
    )  # Ensure within valid range due to numerical errors

    # Calculate the angle in radians
    angle = np.arccos(cos_angle)

    return angle


def calculate_angular_difference(q1, q2):
    """Calculate the angular difference between two quaternions.

    This function computes the absolute angle between two quaternions, representing
    the rotation needed to align the first quaternion with the second one. The result
    is guaranteed to be in the range [0, π], representing the shortest rotation path.

    Parameters:
    q1 (Quaternion): The first quaternion, representing the initial orientation.
    q2 (Quaternion): The second quaternion, representing the target orientation.

    Returns:
    float: The absolute angular difference between the two quaternions in radians.
    """
    # Ensure quaternions are normalized and positive
    q1 = make_quaternion_positive(q1).normalised
    q2 = make_quaternion_positive(q2).normalised

    # Calculate the quaternion representing the rotation from q1 to q2
    q_diff = q1.inverse * q2

    # To account for double coverage of rotation space and singularity around
    # np.pi when determing inverse of quaternions

    # Adjust for quaternion's double-cover property, where each rotation is represented
    # by two quaternions. Handle edge cases near np.pi to accurately compute
    # the inverse, ensuring the shortest rotation path between quaternions is
    # identified, avoiding ambiguities around 180-degree rotations.
    epsilon = 1e-6  # Tolerance level
    dot = np.clip(q_diff.w, -1.0, 1.0)

    if dot < -1 + epsilon:
        angle_diff = np.pi
    else:
        angle_diff = 2 * np.arccos(np.abs(dot))

    return q_diff, angle_diff


def calculate_angular_difference_along_axis(q1, q2, axis=0):
    """Calculate the angular difference along a specified axis between two quaternions.

    This function computes the angle between two vectors obtained by rotating a unit
    vector along the specified axis by the two quaternions. The result represents the
    angular difference around that axis.

    Parameters:
    q1 (array_like): The first quaternion, representing the initial orientation.
    q2 (array_like): The second quaternion, representing the target orientation.
    axis (int): The axis index (0 for x, 1 for y, and 2 for z) to calculate the angular
    difference around.

    Returns:
    float: The angular difference along the specified axis between the two quaternions
    in radians.
    """
    # Define unit vectors for each axis
    unit_vectors = np.identity(3)
    unit_vector = unit_vectors[axis]

    # Convert quaternions to rotation objects
    rot1 = quat2mat(q1)
    rot2 = quat2mat(q2)

    # Rotate the unit vector by each quaternion
    vector1 = np.dot(rot1, unit_vector)
    vector2 = np.dot(rot2, unit_vector)

    # Calculate the cosine of the angle between the two vectors
    cos_angle = np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )
    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0,
    )  # Ensure within valid range due to numerical errors

    # Calculate the angle in radians
    angle = np.arccos(cos_angle)

    return angle
