from abc import ABC, abstractmethod
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.distributions import ParticleDistribution

class BaseGenerator(ABC):
    """
    生成器抽象基类
    """
    def __init__(
        self,
        n_particles: int,
        df: float,
        kf: float,
        particle_dist: ParticleDistribution,
        overlap_tolerance: float = 0.0
    ):
        """
        初始化生成器
        
        Args:
            n_particles (int): 目标粒子数。
            df (float): 分形维数 (Fractal dimension)。
            kf (float): 分形前置因子 (Fractal prefactor)。
            particle_dist (ParticleDistribution): 粒径分布。
            overlap_tolerance (float): 粒子重叠容差。
        """
        self.n_particles = n_particles
        self.df = df
        self.kf = kf
        self.particle_dist = particle_dist
        self.overlap_tolerance = overlap_tolerance

    @abstractmethod
    def generate(self) -> Aggregate:
        """
        执行生成逻辑
        
        Returns:
            Aggregate: 生成的分形团簇。
        """
        pass
