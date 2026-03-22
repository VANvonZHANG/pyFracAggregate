from abc import ABC, abstractmethod
import numpy as np

class ParticleDistribution(ABC):
    """
    粒径分布抽象基类
    """
    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """
        采样生成 n 个粒径。
        
        Args:
            n (int): 需生成的粒子数量。
            
        Returns:
            np.ndarray: 形状为 (n,) 的粒径数组。
        """
        pass

class Monodisperse(ParticleDistribution):
    """
    单分散分布（所有粒子半径相同）。
    """
    def __init__(self, radius: float):
        """
        Args:
            radius (float): 粒子半径。
        Raises:
            ValueError: 若半径 <= 0。
        """
        if radius <= 0:
            raise ValueError("Radius must be positive")
        self.radius = radius
        
    def sample(self, n: int) -> np.ndarray:
        return np.full(n, self.radius, dtype=np.float64)

class LognormalDistribution(ParticleDistribution):
    """
    对数正态分布。
    """
    def __init__(self, mean: float, std: float):
        """
        Args:
            mean (float): 几何均值 (Geometric mean)。
            std (float): 几何标准差 (Geometric standard deviation)。应该 >= 1.0。
            
        Raises:
            ValueError: 若 mean 或 std 不合法。
        """
        if mean <= 0:
            raise ValueError("Mean must be positive")
        if std <= 0:
            raise ValueError("Standard deviation must be positive")
            
        self.mean = mean
        self.std = std
        
        # 计算底层正态分布的均值和标准差
        self.normal_mean = np.log(self.mean)
        self.normal_std = np.log(self.std) if self.std > 1.0 else 0.0
        
    def sample(self, n: int) -> np.ndarray:
        if self.normal_std == 0.0:
            return np.full(n, self.mean, dtype=np.float64)
        return np.random.lognormal(self.normal_mean, self.normal_std, n)
