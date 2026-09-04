from abc import ABC, abstractmethod

import numpy as np
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
        overlap_tolerance: float = 1e-5,
        length_unit: str = 'nm',
        mass_unit: str = 'g',
        density: float = 1.0,
        placement: 'str | PlacementStrategy' = 'solved',
        scaling: 'ScalingLaw | str | None' = None,
        seed: 'int | None' = None,
        surface_beta: float | None = None,
        rng: 'np.random.Generator | None' = None,
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
            placement (str or PlacementStrategy, optional): 'sampled', 'solved',
                or 'constructed' (name or instance). Defaults to 'solved'.
            scaling (ScalingLaw or str, optional): Target-distance law
                ('count'/'mass' name or instance). Defaults to 'mass'.
            seed (int, optional): Seed for reproducible generation.
            surface_beta (float, optional): Surface-particle filter fraction
                (solved placement only). Defaults to 0.3.
            rng (np.random.Generator, optional): Shared generator (used by
                CCA subcluster seeding); overrides seed when given.
        """
        self.n_particles = n_particles
        self.df = df
        self.kf = kf
        self.scaling: ScalingLaw = (
            scaling if isinstance(scaling, ScalingLaw)
            else get_scaling(scaling or "mass", self.df, self.kf)
        )
        self.rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng(seed)
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
            rng=self.rng,
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
