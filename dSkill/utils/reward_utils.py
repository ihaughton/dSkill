"""Reward utils."""

import math

import numpy as np


def long_tail_tolerance(x, lower_bound, upper_bound, margin, value_at_margin=0.1):
    """Top hat distribution with tails"""
    if lower_bound > upper_bound:
        raise Exception("Lower bound must be less than or equal to upperbound")
    value_at_margin = max(
        min(value_at_margin, 1),
        0.0001,
    )  # Clamp value between 0.0001 and 1

    if lower_bound <= x <= upper_bound:
        return 1.0

    if x < lower_bound:
        d = (lower_bound - x) / margin
    else:
        d = (x - upper_bound) / margin

    scale = math.sqrt(1 / value_at_margin - 1)
    return 1 / (math.pow(d * scale, 2) + 1)


def alignment_to_z_axis(p1, p2):
    """
    Calculates the alignment of the vector defined by two 3D points with the z-axis.

    Parameters:
    - p1 (array_like): The first 3D point.
    - p2 (array_like): The second 3D point.

    Returns:
    - float: The alignment with the z-axis. Ranges from 0 to 1.
        - 0 indicates perfect alignment with the z-axis.
        - 1 indicates perpendicular alignment to the z-axis.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)

    V = p2 - p1
    V_magnitude = np.linalg.norm(V)
    V_unit = V / V_magnitude

    Z = np.array([0, 0, 1])
    alignment = np.dot(V_unit, Z)
    alignment = 1 - np.abs(alignment)

    return alignment
