from pyFracAggregate.core.distributions import ParticleDistribution
from pyFracAggregate.generators.base import BaseGenerator
from pyFracAggregate.generators.pca_filippov import PCAFilippovGenerator

def get_generator(
    method: str,
    n_particles: int,
    df: float,
    kf: float,
    particle_dist: ParticleDistribution,
    overlap_tolerance: float = 0.0,
    **kwargs
) -> BaseGenerator:
    """
    获取对应的分形团簇生成器。
    
    Args:
        method (str): 生成算法，例如 'pca', 'cca', 'fracval'。
        n_particles (int): 粒子数量。
        df (float): 分形维数。
        kf (float): 分形前置因子。
        particle_dist (ParticleDistribution): 粒径分布。
        overlap_tolerance (float): 重叠容差。
        **kwargs: 其它算法特定参数。
        
    Returns:
        BaseGenerator: 生成器实例。
        
    Raises:
        ValueError: 若 method 不被支持。
    """
    method = method.lower()
    if method == 'pca':
        return PCAFilippovGenerator(n_particles, df, kf, particle_dist, overlap_tolerance)
    else:
        raise ValueError(f"Unknown generation method: {method}")
