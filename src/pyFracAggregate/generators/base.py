from abc import ABC, abstractmethod
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.distributions import ParticleDistribution

class BaseGenerator(ABC):
    """Abstract base class for generators."""
    def __init__(
        self,
        n_particles: int,
        df: float,
        kf: float,
        particle_dist: ParticleDistribution,
        overlap_tolerance: float = 0.0
    ):
        """Initializes the generator.
        
        Args:
            n_particles (int): Target number of particles.
            df (float): Fractal dimension.
            kf (float): Fractal prefactor.
            particle_dist (ParticleDistribution): Particle size distribution.
            overlap_tolerance (float): Particle overlap tolerance.
        """
        self.n_particles = n_particles
        self.df = df
        self.kf = kf
        self.particle_dist = particle_dist
        self.overlap_tolerance = overlap_tolerance

    @abstractmethod
    def generate(self) -> Aggregate:
        """Executes the generation logic.
        
        Returns:
            Aggregate: The generated fractal cluster.
        """
        pass
