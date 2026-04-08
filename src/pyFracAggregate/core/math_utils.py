import numpy as np
from typing import Tuple, Optional

try:
    import mathutils
    HAS_MATHUTILS = True
except ImportError:
    HAS_MATHUTILS = False
    from scipy.spatial.transform import Rotation as R

def rotate_points(points: np.ndarray, euler_angles: Tuple[float, float, float]) -> np.ndarray:
    """
    Rotate points using given euler angles (in radians).
    
    Args:
        points (np.ndarray): Shape (N, 3) points to rotate.
        euler_angles (Tuple[float, float, float]): Rotation angles (x, y, z) in radians.
        
    Returns:
        np.ndarray: Rotated points of shape (N, 3).
    """
    if points.size == 0:
        return points.copy()
        
    if HAS_MATHUTILS:
        # Create an Euler rotation
        euler = mathutils.Euler(euler_angles, 'XYZ')
        # Convert to a 3x3 rotation matrix
        rot_matrix = np.array(euler.to_matrix())
        # Apply rotation
        return points @ rot_matrix.T
    else:
        # Fallback for Python 3.12 where mathutils might fail to install
        rot = R.from_euler('XYZ', euler_angles, degrees=False)
        return rot.apply(points)

def rotate_points_quaternion(points: np.ndarray, quaternion: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Rotate points using given quaternion (w, x, y, z).
    
    Args:
        points (np.ndarray): Shape (N, 3) points to rotate.
        quaternion (Tuple[float, float, float, float]): Quaternion parameters (w, x, y, z).
        
    Returns:
        np.ndarray: Rotated points.
    """
    if points.size == 0:
        return points.copy()
        
    if HAS_MATHUTILS:
        q = mathutils.Quaternion(quaternion)
        rot_matrix = np.array(q.to_matrix())
        return points @ rot_matrix.T
    else:
        # scipy Rotation uses (x, y, z, w) instead of mathutils' (w, x, y, z)
        w, x, y, z = quaternion
        rot = R.from_quat([x, y, z, w])
        return rot.apply(points)

def euler_rodrigues_rotation(points: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate points around an arbitrary axis by a given angle using Euler-Rodrigues formula.

    Args:
        points (np.ndarray): Shape (N, 3) points to rotate.
        axis (np.ndarray): Rotation axis (3,), must be non-zero.
        angle (float): Rotation angle in radians.

    Returns:
        np.ndarray: Rotated points of shape (N, 3).
    """
    if points.size == 0:
        return points.copy()
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return points @ R.T


def sphere_sphere_intersection(
    c1: np.ndarray, r1: float, c2: np.ndarray, r2: float
) -> Optional[Tuple[np.ndarray, float]]:
    """Compute the intersection circle of two spheres.

    Args:
        c1: Center of sphere 1, shape (3,).
        r1: Radius of sphere 1.
        c2: Center of sphere 2, shape (3,).
        r2: Radius of sphere 2.

    Returns:
        (circle_center, circle_radius) if intersection exists, None otherwise.
        For tangent spheres, circle_radius is 0.0.
    """
    d_vec = c2 - c1
    d = np.linalg.norm(d_vec)

    if d > r1 + r2 + 1e-12:
        return None
    if d < abs(r1 - r2) - 1e-12:
        return None
    if d < 1e-12:
        return None

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h_sq = r1**2 - a**2
    if h_sq < -1e-12:
        return None
    h = np.sqrt(max(h_sq, 0.0))

    circle_center = c1 + (a / d) * d_vec
    return circle_center, h


def random_point_on_circle(
    center: np.ndarray,
    radius: float,
    normal: np.ndarray,
) -> np.ndarray:
    """Sample a random point on a circle in 3D.

    Args:
        center: Circle center, shape (3,).
        radius: Circle radius.
        normal: Normal vector to the circle plane, shape (3,).

    Returns:
        A single point on the circle, shape (3,).
    """
    normal = normal / np.linalg.norm(normal)
    temp = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(normal, temp)) > 0.9:
        temp = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, temp)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    theta = np.random.uniform(0, 2 * np.pi)
    return center + radius * (np.cos(theta) * u + np.sin(theta) * v)
