import numpy as np
from scipy.spatial import cKDTree
from pyFracAggregate.core.aggregate import Aggregate

def pair_correlation_function(
    aggregate: Aggregate, 
    bins: int = 50, 
    r_max: float = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算两点密度相关函数 C(r) (Two-point density correlation function)。
    基于 scipy.spatial.cKDTree 提高计算效率。
    
    C(r) = n(r) / (4 * pi * r^2 * h * N)
    其中 n(r) 是距离为 r 到 r+h 之间的粒子对(pair)数量，N 是总粒子数，h 是步长 (bin width)。
    
    Args:
        aggregate (Aggregate): 团簇对象。
        bins (int): r 的分箱数。
        r_max (float, optional): 计算相关函数的最大距离。如果为 None，则取粒子间最大距离。
        
    Returns:
        tuple[np.ndarray, np.ndarray]: (r_centers, C_r) 分别是距离的中心值和对应的相关函数值。
    """
    if aggregate.current_size < 2:
        return np.array([]), np.array([])
        
    positions = aggregate.positions
    N = aggregate.current_size
    
    # 构建 KDTree 以加速查询
    tree = cKDTree(positions)
    
    if r_max is None:
        # 估算最大距离 (最远两点距离的上限)
        # 用中心到边缘的最大距离的 2 倍作为安全的上界
        com = np.mean(positions, axis=0)
        max_dist_to_center = np.max(np.linalg.norm(positions - com, axis=1))
        r_max = 2.0 * max_dist_to_center
        if r_max == 0:
            return np.array([]), np.array([])
            
    # 计算距离统计 (只取上三角，避免重复计算，所以对数需要 x2)
    # tree.count_neighbors 返回的是累计数量 (<= r)，所以要进行差分
    r_edges = np.linspace(0, r_max, bins + 1)
    
    # count_neighbors 对于多个半径，可以一次性传入
    # cumulative_counts[i] 包含距离 <= r_edges[i] 的对数
    cumulative_counts = tree.count_neighbors(tree, r_edges)
    
    # 因为 r=0 时包含对自身的 N 次匹配。我们需要先把 N 减去，避免干扰真实的成对统计
    # (如果两点碰巧重合，距离也为 0，这在我们的应用中是不应该存在的，因为做了碰撞检测)
    cumulative_counts = np.array(cumulative_counts, dtype=np.float64)
    # 对于所有的 r>=0 的计数，我们都减去自匹配的 N 个
    cumulative_counts -= N
    # 确保没有负数（防止因为精度问题导致的异常）
    cumulative_counts = np.maximum(cumulative_counts, 0)
    
    # 差分得到每个 bin 内的数量: n(r)
    n_r = np.diff(cumulative_counts)
    
    # r_centers 取区间中点
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2.0
    h = r_edges[1] - r_edges[0]
    
    # 避免 r=0 除以 0 的情况，虽然 r_centers > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        c_r = n_r / (4.0 * np.pi * (r_centers ** 2) * h * N)
        
    # 对于距离为 0 的情况，或者由于距离太小除以 0，赋为 0
    c_r = np.nan_to_num(c_r, posinf=0.0)
    
    return r_centers, c_r
