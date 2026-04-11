import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.generators.factory import get_generator
from pyFracAggregate.core.distributions import Monodisperse, LognormalDistribution
from pyFracAggregate.analysis.morphology import radius_of_gyration, center_of_mass
from pyFracAggregate.analysis.correlation import (
    pair_correlation_function,
    estimate_fractal_dimension,
    plot_pair_correlation
)
from pyFracAggregate.io.vtk import export_vtm, export_vtk
from pyFracAggregate.io.data import export_yaml
from pyFracAggregate.generators.pca import PCAGenerator
from pyFracAggregate.generators.cca import CCAGenerator
from pyFracAggregate.generators.fracval import FracVALGenerator
from pyFracAggregate.generators.tdcca import ThouyJullienGenerator
from pyFracAggregate.generators.placement.algebraic import AlgebraicPlacement
from pyFracAggregate.generators.placement.random_ import RandomPlacement

__version__ = "0.1.0"
__author__ = "Fan Zhang"

def generate(
    n_particles: int,
    df: float,
    kf: float,
    method: str = 'pca',
    particle_dist = None,
    overlap_tolerance: float = 1e-5,
    placement: str = 'algebraic',
    **kwargs
) -> Aggregate:
    """
    High-level API to generate a fractal aggregate.

    Args:
        n_particles (int): Target number of particles.
        df (float): Fractal dimension (typically 1.5 - 2.5).
        kf (float): Fractal prefactor (typically 1.0 - 2.0).
        method (str): Algorithm to use ('pca', 'cca', 'fracval', 'tdcca').
        particle_dist: Particle radius distribution (defaults to Monodisperse(1.0)).
        overlap_tolerance (float): Allowed overlap between spheres.
        placement (str): Placement strategy ('algebraic' or 'random').

    Returns:
        Aggregate: The generated fractal aggregate.
    """
    if particle_dist is None:
        particle_dist = Monodisperse(1.0)

    generator = get_generator(
        method=method,
        n_particles=n_particles,
        df=df,
        kf=kf,
        particle_dist=particle_dist,
        overlap_tolerance=overlap_tolerance,
        placement=placement,
        **kwargs
    )

    return generator.generate()

def analyze(aggregate: Aggregate):
    """
    Compute core morphological properties.
    """
    rg = radius_of_gyration(aggregate)
    r_centers, c_r = pair_correlation_function(aggregate)
    df_est, r2, _ = estimate_fractal_dimension(r_centers, c_r,
                                               r_min=np.mean(aggregate.radii),
                                               r_max=rg)

    return {
        "Rg": rg,
        "CoM": center_of_mass(aggregate),
        "N": aggregate.current_size,
        "Df_estimated": df_est,
        "R2": r2
    }

__all__ = [
    "generate",
    "analyze",
    "Aggregate",
    "Monodisperse",
    "LognormalDistribution",
    "radius_of_gyration",
    "center_of_mass",
    "pair_correlation_function",
    "estimate_fractal_dimension",
    "plot_pair_correlation",
    "export_yaml",
    "export_vtm",
    "export_vtk",
    "PCAGenerator",
    "CCAGenerator",
    "FracVALGenerator",
    "ThouyJullienGenerator",
    "AlgebraicPlacement",
    "RandomPlacement",
]
