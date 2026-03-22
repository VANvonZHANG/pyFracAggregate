import pytest
import numpy as np
from pyFracAggregate.core.math_utils import rotate_points, rotate_points_quaternion

def test_rotate_points_distance_invariance():
    points = np.array([
        [1.0, 2.0, 3.0],
        [-1.0, 5.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    
    # Calculate original pairwise distances
    dist_orig_0_1 = np.linalg.norm(points[0] - points[1])
    dist_orig_0_2 = np.linalg.norm(points[0] - points[2])
    
    euler_angles = (np.pi/4, np.pi/3, -np.pi/6)
    rotated = rotate_points(points, euler_angles)
    
    # Calculate new pairwise distances
    dist_rot_0_1 = np.linalg.norm(rotated[0] - rotated[1])
    dist_rot_0_2 = np.linalg.norm(rotated[0] - rotated[2])
    
    # Verify distance invariance (error < 1e-6)
    assert abs(dist_orig_0_1 - dist_rot_0_1) < 1e-6
    assert abs(dist_orig_0_2 - dist_rot_0_2) < 1e-6
    
    # Center (0,0,0) should remain (0,0,0)
    np.testing.assert_allclose(rotated[2], [0.0, 0.0, 0.0], atol=1e-6)

def test_rotate_points_quaternion_distance_invariance():
    points = np.array([
        [5.0, -2.0, 1.0],
        [0.0, 3.0, -1.0]
    ])
    
    dist_orig = np.linalg.norm(points[0] - points[1])
    
    # w, x, y, z
    quaternion = (0.5, 0.5, 0.5, 0.5) 
    rotated = rotate_points_quaternion(points, quaternion)
    
    dist_rot = np.linalg.norm(rotated[0] - rotated[1])
    
    assert abs(dist_orig - dist_rot) < 1e-6
