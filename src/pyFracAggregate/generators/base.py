from abc import ABC, abstractmethod
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.distributions import ParticleDistribution
from pyFracAggregate.core.scaling import ScalingLaw, get_scaling
from pyFracAggregate.generators.placement.base import PlacementStrategy, get_placement


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
        density: float = 1.0,
        placement: str = 'solved',
        scaling: ScalingLaw | None = None,
        surface_beta: float | None = None,
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
            placement (str, optional): Placement strategy name ('sampled', 'solved',
                or 'constructed'). Defaults to 'solved'.
            scaling (ScalingLaw, optional): Target-distance law. Defaults to CountScaling.
            surface_beta (float, optional): Surface-particle filter fraction
                (solved placement only). Defaults to 0.3.
        """
        self.n_particles = n_particles
        self.df = df
        self.kf = kf
        self.scaling: ScalingLaw = (
            scaling if isinstance(scaling, ScalingLaw)
            else get_scaling(scaling or "count", self.df, self.kf)
        )
        self.particle_dist = particle_dist
        self.overlap_tolerance = overlap_tolerance
        self.length_unit = length_unit
        self.mass_unit = mass_unit
        self.density = density
        self.placement: PlacementStrategy = get_placement(
            placement,
            overlap_tolerance=overlap_tolerance,
            surface_beta=surface_beta if surface_beta is not None else 0.3,
        )
        self._resolved_placement = {
            "SolvedPlacement": "solved",
            "SampledPlacement": "sampled",
            "ConstructedPlacement": "constructed",
        }[type(self.placement).__name__]

    @abstractmethod
    def generate(self) -> Aggregate:
        """Executes the generation logic.
        
        Returns:
            Aggregate: The generated fractal cluster.
        """
        pass
