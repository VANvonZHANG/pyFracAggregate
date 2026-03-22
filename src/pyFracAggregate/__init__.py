"""pyFracAggregate core package."""

from typing import Optional

from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.distributions import Monodisperse, LognormalDistribution, ParticleDistribution
from pyFracAggregate.generators.factory import get_generator
import pyFracAggregate.analysis as analyze

__all__ = [
    "Aggregate",
    "Monodisperse",
    "LognormalDistribution",
    "ParticleDistribution",
    "generate",
    "analyze"
]

def generate(
    n_particles: int, 
    df: float, 
    kf: float, 
    method: str = 'pca',
    optimization: str = 'monte_carlo',
    particle_dist: Optional[ParticleDistribution] = None,
    overlap_tolerance: float = 0.0,
    **kwargs
) -> Aggregate:
    """
    统一的顶层分形团簇生成接口。
    
    Args:
        n_particles (int): 团簇包含的粒子总数。
        df (float): 分形维数 (Fractal Dimension, 1.0 < df <= 3.0)。
        kf (float): 分形前置因子 (Fractal Prefactor)。
        method (str): 核心生成逻辑，可选 'pca', 'cca', 'fracval'。
        optimization (str): 底层加速引擎，可选 'monte_carlo' 或 'flage'。
        particle_dist (ParticleDistribution, optional): 粒径分布。默认单分散(半径1.0)。
        overlap_tolerance (float): 允许的粒子重叠深度。默认 0.0。
        **kwargs: 其它参数，会向下传递给特定算法。
        
    Returns:
        Aggregate: 生成完毕并填充坐标的团簇对象。
        
    Raises:
        ValueError: 若参数不合法 (如 df > 3.0)。
    """
    if df <= 0.0 or df > 3.0:
        raise ValueError("Fractal dimension df must be in (0, 3.0]")
        
    if particle_dist is None:
        particle_dist = Monodisperse(1.0)
        
    generator = get_generator(
        method=method,
        n_particles=n_particles,
        df=df,
        kf=kf,
        particle_dist=particle_dist,
        overlap_tolerance=overlap_tolerance,
        optimization=optimization,
        **kwargs
    )
    
    return generator.generate()
