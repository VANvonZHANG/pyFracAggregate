from pyFracAggregate.core.distributions import ParticleDistribution
from pyFracAggregate.generators.base import BaseGenerator
from pyFracAggregate.generators.pca import PCAGenerator
from pyFracAggregate.generators.cca import CCAGenerator
from pyFracAggregate.generators.fracval import FracVALGenerator
from pyFracAggregate.generators.tdcca import ThouyJullienGenerator


def get_generator(
    method: str,
    n_particles: int,
    df: float,
    kf: float,
    particle_dist: ParticleDistribution,
    overlap_tolerance: float = 0.0,
    **kwargs
) -> BaseGenerator:
    method = method.lower()

    if method == 'pca':
        return PCAGenerator(n_particles, df, kf, particle_dist, overlap_tolerance, **kwargs)
    elif method == 'cca':
        return CCAGenerator(n_particles, df, kf, particle_dist, overlap_tolerance, **kwargs)
    elif method == 'fracval':
        return FracVALGenerator(n_particles, df, kf, particle_dist, overlap_tolerance, **kwargs)
    elif method == 'tdcca':
        return ThouyJullienGenerator(n_particles, df, kf, particle_dist, overlap_tolerance, **kwargs)
    elif method in ('flage_pca', 'flage_cca'):
        raise ValueError(
            f"method='{method}' has been removed. Use method='{'pca' if method == 'flage_pca' else 'cca'}' "
            "(FLAGE is now the default placement strategy). "
            f"For the old random sampling behavior, use method='{'pca' if method == 'flage_pca' else 'cca'}', placement='random'."
        )
    else:
        raise ValueError(f"Unknown generation method: {method}")
