import numpy as np
from typing import Tuple, List

def find_exact_touching_points_pca(
    center: np.ndarray,
    L: float,
    ref_pos: np.ndarray,
    r_new: float,
    r_ref: float,
    num_points: int = 8
) -> np.ndarray:
    """
    基于 Skorupski 等 (2014) FLAGE 算法的解析几何求解器（适用于 PCA）。
    计算新粒子在距离中心为 L 的球面上，同时恰好与参考粒子相切的位置。
    
    这相当于求两个球面的交线：
    1. 以 center 为球心，半径为 L 的球面。
    2. 以 ref_pos 为球心，半径为 r_new + r_ref 的球面。
    
    Args:
        center (np.ndarray): 团簇质心/几何中心。
        L (float): 新粒子与中心的距离限制。
        ref_pos (np.ndarray): 参考粒子的坐标。
        r_new (float): 新粒子的半径。
        r_ref (float): 参考粒子的半径。
        num_points (int): 在交线圆上采样的点数。
        
    Returns:
        np.ndarray: 形状为 (K, 3) 的有效坐标点数组 (可能为空，如果两个球面不相交)。
    """
    # 向量 C -> B (中心到参考粒子)
    CB = ref_pos - center
    dist_CB = np.linalg.norm(CB)
    
    if dist_CB < 1e-8:
        # 参考粒子在中心，交线不再是圆，而是如果 L == r_new + r_ref 则是整个球面
        # 此处不处理这种极端退化情况
        return np.empty((0, 3))
        
    # 余弦定理计算角度 alpha
    # 三角形 CAB 中:
    # |CA| = L
    # |CB| = dist_CB
    # |AB| = r_new + r_ref
    dist_AB = r_new + r_ref
    
    cos_alpha = (L**2 + dist_CB**2 - dist_AB**2) / (2 * L * dist_CB)
    
    if cos_alpha < -1.0 or cos_alpha > 1.0:
        # 两个球面不相交
        return np.empty((0, 3))
        
    alpha = np.arccos(cos_alpha)
    
    # 构建一个正交基，其中 u 平行于 CB
    u = CB / dist_CB
    
    # 找一个与 u 正交的随机向量 v
    temp = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(u, temp)) > 0.9:
        temp = np.array([0.0, 1.0, 0.0])
    v = np.cross(u, temp)
    v /= np.linalg.norm(v)
    
    # w 与 u, v 构成右手正交基
    w = np.cross(u, v)
    
    # 圆的半径和中心
    # 新粒子 A 的投影点在 CB 上的长度为 L * cos_alpha
    circle_center = center + u * (L * cos_alpha)
    circle_radius = L * np.sin(alpha)
    
    # 在圆上采样 num_points 个点
    thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    points = np.zeros((num_points, 3))
    for i, theta in enumerate(thetas):
        points[i] = circle_center + circle_radius * (np.cos(theta) * v + np.sin(theta) * w)
        
    return points

def filter_overlapping_candidates(
    candidates: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    r_new: float,
    overlap_tolerance: float = 1e-5
) -> np.ndarray:
    """
    过滤掉发生重叠的候选点。
    """
    if len(candidates) == 0:
        return candidates
        
    valid_candidates = []
    # min_dists 广播
    min_dists = radii + r_new - overlap_tolerance
    
    for cand in candidates:
        dists = np.linalg.norm(positions - cand, axis=1)
        if not np.any(dists < min_dists):
            valid_candidates.append(cand)
            
    return np.array(valid_candidates)
