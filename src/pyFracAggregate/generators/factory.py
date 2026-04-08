from pyFracAggregate.core.distributions import ParticleDistribution
from pyFracAggregate.generators.base import BaseGenerator
from pyFracAggregate.generators.pca_filippov import PCAFilippovGenerator
from pyFracAggregate.generators.cca_filippov import CCAFilippovGenerator
from pyFracAggregate.generators.cca_fracval import FracVALGenerator
from pyFracAggregate.generators.pca_flage import PCAFlageGenerator
from pyFracAggregate.generators.cca_flage import CCAFlageGenerator

def get_generator(
    method: str,
    n_particles: int,
    df: float,
    kf: float,
    particle_dist: ParticleDistribution,
    overlap_tolerance: float = 0.0,
    **kwargs
) -> BaseGenerator:
    """Gets the corresponding fractal cluster generator.
    
    Args:
        method (str): Generation algorithm ('pca', 'cca', 'fracval').
        n_particles (int): Number of particles.
        df (float): Fractal dimension.
        kf (float): Fractal prefactor.
        particle_dist (ParticleDistribution): Particle size distribution.
        overlap_tolerance (float): Overlap tolerance.
        **kwargs: Additional algorithm-specific parameters.
        
    Returns:
        BaseGenerator: Generator instance.
        
    Raises:
        ValueError: If method is not supported.
    """
    method = method.lower()
    if method == 'pca':
        return PCAFilippovGenerator(n_particles, df, kf, particle_dist, overlap_tolerance, **kwargs)
    elif method == 'cca':
        return CCAFilippovGenerator(n_particles, df, kf, particle_dist, overlap_tolerance, **kwargs)
    elif method == 'fracval':
        return FracVALGenerator(n_particles, df, kf, particle_dist, overlap_tolerance, **kwargs)
    elif method == 'flage_pca':
        return PCAFlageGenerator(n_particles, df, kf, particle_dist, overlap_tolerance, **kwargs)
    elif method == 'flage_cca':
        return CCAFlageGenerator(n_particles, df, kf, particle_dist, overlap_tolerance, **kwargs)
    else:
        raise ValueError(f"Unknown generation method: {method}")
