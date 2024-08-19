import unittest
import numpy as np
from geom import Transform3D, Quaternion, degrees_to_radians, radians_to_degrees

class TestTransform3D(unittest.TestCase):

    def test_as_matrix(self):
        translation = np.array([1, 2, 3])
        quaternion = Quaternion(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        transform = Transform3D(translation, quaternion)
        expected_matrix = np.array([
            [1, 0, 0, 1],
            [0, 0, -1, 2],
            [0, 1, 0, 3],
            [0, 0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(transform.as_matrix, expected_matrix, decimal=4)

    def test_from_matrix(self):
        matrix = np.array([
            [1, 0, 0, 1],
            [0, 0, -1, 2],
            [0, 1, 0, 3],
            [0, 0, 0, 1]
        ])
        transform = Transform3D.from_matrix(matrix)
        np.testing.assert_array_almost_equal(transform.translation, [1, 2, 3])
        np.testing.assert_array_almost_equal(transform.quaternion, [0.7071, 0.7071, 0, 0], decimal=4)

    def test_mul(self):
        translation1 = np.array([1, 2, 3])
        quaternion1 = Quaternion(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        transform1 = Transform3D(translation1, quaternion1)

        translation2 = np.array([4, 5, 6])
        quaternion2 = Quaternion(0.7071, 0, 0.7071, 0)  # 90 degrees rotation around y-axis
        transform2 = Transform3D(translation2, quaternion2)

        result = transform1 * transform2
        expected_translation = np.array([5, -4, 8])
        expected_quaternion = Quaternion(0.5, 0.5, 0.5, 0.5)  # Combined rotation
        np.testing.assert_array_almost_equal(result.translation, expected_translation, decimal=4)
        np.testing.assert_array_almost_equal(result.quaternion, expected_quaternion, decimal=4)

    def test_call(self):
        translation = np.array([1, 2, 3])
        quaternion = Quaternion(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        transform = Transform3D(translation, quaternion)
        vector = np.array([1, 1, 1])
        result = transform(vector)
        expected_result = np.array([2, 1, 4])
        np.testing.assert_array_almost_equal(result, expected_result, decimal=4)

class TestQuaternion(unittest.TestCase):

    def test_mul(self):
        q1 = Quaternion(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        q2 = Quaternion(0.7071, 0, 0.7071, 0)  # 90 degrees rotation around y-axis
        result = q1 * q2
        expected_result = Quaternion(0.5, 0.5, 0.5, 0.5)  # Combined rotation
        np.testing.assert_array_almost_equal(result, expected_result, decimal=4)

    def test_call(self):
        q = Quaternion(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        vector = np.array([1, 1, 1])
        result = q(vector)
        expected_result = np.array([1, -1, 1])
        np.testing.assert_array_almost_equal(result, expected_result, decimal=4)

    def test_inv(self):
        q = Quaternion(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        result = q.inv
        expected_result = Quaternion(0.7071, -0.7071, 0, 0)
        np.testing.assert_array_almost_equal(result, expected_result, decimal=4)

    def test_as_rotation_matrix(self):
        q = Quaternion(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        expected_matrix = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        np.testing.assert_array_almost_equal(q.as_rotation_matrix, expected_matrix, decimal=4)

    def test_from_rotation_matrix(self):
        matrix = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        q = Quaternion.from_rotation_matrix(matrix)
        expected_result = Quaternion(0.7071, 0.7071, 0, 0)
        np.testing.assert_array_almost_equal(q, expected_result, decimal=4)

    def test_euler_to_quat(self):
        euler = np.array([np.pi / 2, 0, 0])  # 90 degrees rotation around x-axis
        q = Quaternion.from_euler(euler)
        expected_result = Quaternion(0.7071, 0.7071, 0, 0)
        np.testing.assert_array_almost_equal(q, expected_result, decimal=4)

    def test_quat_to_euler(self):
        q = Quaternion(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        euler = q.as_euler
        expected_result = np.array([np.pi / 2, 0, 0])
        np.testing.assert_array_almost_equal(euler, expected_result, decimal=4)

class TestUtils(unittest.TestCase):

    def test_degrees_to_radians(self):
        degrees = 180
        radians = degrees_to_radians(degrees)
        self.assertAlmostEqual(radians, np.pi)

    def test_radians_to_degrees(self):
        radians = np.pi
        degrees = radians_to_degrees(radians)
        self.assertAlmostEqual(degrees, 180)

if __name__ == '__main__':
    unittest.main()
