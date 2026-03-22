import numpy as np
from typing import Tuple

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
