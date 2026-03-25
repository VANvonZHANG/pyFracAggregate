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
        overlap_tolerance: float = 0.0,
        length_unit: str = 'nm',
        mass_unit: str = 'g',
        density: float = 1.0
    ):
        """Initializes the generator.
        
        Args:
            n_particles (int): Target number of particles.
            df (float): Fractal dimension.
            kf (float): Fractal prefactor.
            particle_dist (ParticleDistribution): Particle size distribution.
            overlap_tolerance (float): Particle overlap tolerance.
            length_unit (str, optional): Unit for length. Defaults to 'nm'.
            mass_unit (str, optional): Unit for mass. Defaults to 'g'.
            density (float, optional): Density of particle material. Defaults to 1.0.
        """
        self.n_particles = n_particles
        self.df = df
        self.kf = kf
        self.particle_dist = particle_dist
        self.overlap_tolerance = overlap_tolerance
        self.length_unit = length_unit
        self.mass_unit = mass_unit
        self.density = density

    @abstractmethod
    def generate(self) -> Aggregate:
        """Executes the generation logic.
        
        Returns:
            Aggregate: The generated fractal cluster.
        """
        pass
