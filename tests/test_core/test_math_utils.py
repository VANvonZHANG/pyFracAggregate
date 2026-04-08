import pytest
import numpy as np
from pyFracAggregate.core.math_utils import rotate_points, rotate_points_quaternion, euler_rodrigues_rotation, sphere_sphere_intersection

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

def test_euler_rodrigues_rotation_identity():
    """Rotating by zero angle returns original points."""
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    axis = np.array([0.0, 0.0, 1.0])
    rotated = euler_rodrigues_rotation(points, axis, 0.0)
    np.testing.assert_allclose(rotated, points, atol=1e-10)

def test_euler_rodrigues_rotation_90deg_z():
    """Rotating (1,0,0) by 90 degrees around z-axis gives (0,1,0)."""
    points = np.array([[1.0, 0.0, 0.0]])
    axis = np.array([0.0, 0.0, 1.0])
    rotated = euler_rodrigues_rotation(points, axis, np.pi / 2)
    np.testing.assert_allclose(rotated, [[0.0, 1.0, 0.0]], atol=1e-10)

def test_euler_rodrigues_rotation_180deg_x():
    """Rotating (0,1,0) by 180 degrees around x-axis gives (0,-1,0)."""
    points = np.array([[0.0, 1.0, 0.0]])
    axis = np.array([1.0, 0.0, 0.0])
    rotated = euler_rodrigues_rotation(points, axis, np.pi)
    np.testing.assert_allclose(rotated, [[0.0, -1.0, 0.0]], atol=1e-10)

def test_sphere_sphere_intersection_standard():
    """Two spheres of radius 2 centered at (0,0,0) and (3,0,0).
    Intersection circle: center (1.5,0,0), radius = sqrt(4 - 2.25) = sqrt(1.75)."""
    c1, r1 = np.array([0.0, 0.0, 0.0]), 2.0
    c2, r2 = np.array([3.0, 0.0, 0.0]), 2.0
    circle_center, circle_radius = sphere_sphere_intersection(c1, r1, c2, r2)
    np.testing.assert_allclose(circle_center, [1.5, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(circle_radius, np.sqrt(1.75), atol=1e-10)

def test_sphere_sphere_intersection_tangent():
    """Tangent spheres: intersection is a single point."""
    c1, r1 = np.array([0.0, 0.0, 0.0]), 2.0
    c2, r2 = np.array([4.0, 0.0, 0.0]), 2.0
    circle_center, circle_radius = sphere_sphere_intersection(c1, r1, c2, r2)
    np.testing.assert_allclose(circle_center, [2.0, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(circle_radius, 0.0, atol=1e-10)

def test_sphere_sphere_intersection_no_intersection():
    """Non-intersecting spheres return None."""
    c1, r1 = np.array([0.0, 0.0, 0.0]), 2.0
    c2, r2 = np.array([10.0, 0.0, 0.0]), 2.0
    result = sphere_sphere_intersection(c1, r1, c2, r2)
    assert result is None
