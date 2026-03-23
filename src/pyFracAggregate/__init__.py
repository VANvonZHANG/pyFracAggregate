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
    """Unified top-level interface for fractal cluster generation.
    
    Args:
        n_particles (int): Total number of particles in the cluster.
        df (float): Fractal dimension (1.0 < df <= 3.0).
        kf (float): Fractal prefactor.
        method (str): Core generation logic ('pca', 'cca', 'fracval').
        optimization (str): Underlying acceleration engine ('monte_carlo' or 'flage').
        particle_dist (ParticleDistribution, optional): Particle size distribution.
            Defaults to Monodisperse(1.0).
        overlap_tolerance (float): Allowed overlap depth between particles. Defaults to 0.0.
        **kwargs: Additional parameters passed to the specific algorithm.
        
    Returns:
        Aggregate: Generated cluster object with filled coordinates.
        
    Raises:
        ValueError: If parameters are invalid (e.g., df > 3.0).
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
