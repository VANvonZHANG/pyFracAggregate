from abc import ABC, abstractmethod
import numpy as np

class ParticleDistribution(ABC):
    """Abstract base class for particle size distributions."""
    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Samples n particle sizes.
        
        Args:
            n (int): Number of particle sizes to generate.
            
        Returns:
            np.ndarray: An array of particle sizes with shape (n,).
        """
        pass

class Monodisperse(ParticleDistribution):
    """Monodisperse distribution (all particles have the same radius)."""
    def __init__(self, radius: float):
        """Initializes the distribution.

        Args:
            radius (float): Particle radius.
        Raises:
            ValueError: If radius <= 0.
        """
        if radius <= 0:
            raise ValueError("Radius must be positive")
        self.radius = radius
        
    def sample(self, n: int) -> np.ndarray:
        return np.full(n, self.radius, dtype=np.float64)

class LognormalDistribution(ParticleDistribution):
    """Lognormal distribution."""
    def __init__(self, mean: float, std: float):
        """Initializes the distribution.

        Args:
            mean (float): Geometric mean.
            std (float): Geometric standard deviation (should be >= 1.0).
            
        Raises:
            ValueError: If mean or std is invalid.
        """
        if mean <= 0:
            raise ValueError("Mean must be positive")
        if std <= 0:
            raise ValueError("Standard deviation must be positive")
            
        self.mean = mean
        self.std = std
        
        # Calculate mean and standard deviation of the underlying normal distribution
        self.normal_mean = np.log(self.mean)
        self.normal_std = np.log(self.std) if self.std > 1.0 else 0.0
        
    def sample(self, n: int) -> np.ndarray:
        if self.normal_std == 0.0:
            return np.full(n, self.mean, dtype=np.float64)
        return np.random.lognormal(self.normal_mean, self.normal_std, n)
