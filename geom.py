from collections import namedtuple

import numpy as np

# y = R * x + t
Transform3D = namedtuple('Transform3D', ['translation', 'quaternion'])

# All quaternions are in the form [w, x, y, z]
def q_mul(q1, q2):
    """
    Multiply two quaternions. y = R1 * R2 * x
    """
    return np.concatenate(([q1[0] * q2[0] - np.dot(q1[1:], q2[1:])],
                           q1[0] * q2[1:] + q2[0] * q1[1:] + np.cross(q1[1:], q2[1:])))

def q_inv(q):
    """
    Invert a quaternion.
    """
    return np.concatenate(([q[0]], -q[1:]))

def euler_to_quat(euler):
    """
    Convert euler angles in radians to a quaternion.
    """
    roll, pitch, yaw = euler
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])

def quat_to_euler(q):
    """
    Convert a quaternion to euler angles in radians.
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3])
    cosr_cosp = 1 - 2 * (q[1]**2 + q[2]**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (q[0] * q[2] - q[3] * q[1])
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = 1 - 2 * (q[2]**2 + q[3]**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

def degrees_to_radians(degrees):
    """
    Convert degrees to radians.
    """
    return degrees * np.pi / 180.0

def radians_to_degrees(radians):
    """
    Convert radians to degrees.
    """
    return radians * 180.0 / np.pi