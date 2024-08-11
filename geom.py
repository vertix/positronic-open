from collections import namedtuple

import numpy as np

# y = R * x + t
Transform = namedtuple('Transform', ['translation', 'quaternion'])

# All quaternions are in the form [w, x, y, z]
def q_mul(q1, q2):
    """
    Multiply two quaternions. y = R1 * R2 * x
    """
    return np.array([q1[0] * q2[0] - np.dot(q1[1:], q2[1:]),
                     q1[0] * q2[1:] + q2[0] * q1[1:] + np.cross(q1[1:], q2[1:])])

def q_inv(q):
    """
    Invert a quaternion.
    """
    return np.array([q[0], -q[1:]])
