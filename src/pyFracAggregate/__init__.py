"""pyFracAggregate: generation and analysis of synthetic fractal aggregates.

This library builds clusters of spherical primary particles (soot-like,
aerosol-like) with tunable fractal morphology, unifying the classical
generation algorithm families — particle-cluster aggregation (PCA) and
cluster-cluster aggregation (CCA) — on three orthogonal axes: method,
scaling (count/mass), and placement (sampled/solved/constructed).
It also provides morphological analysis (radius of gyration, pair correlation
function, fractal-dimension estimation) and export to YAML, VTK/VTM, rendered
images, and rotation videos.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.distributions import ParticleDistribution
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
from pyFracAggregate.io.visualization import export_render, export_rotation_video
from pyFracAggregate.generators.pca import PCAGenerator
from pyFracAggregate.generators.cca import CCAGenerator
from pyFracAggregate.generators.placement.solved import SolvedPlacement
from pyFracAggregate.generators.placement.sampled import SampledPlacement
from pyFracAggregate.generators.placement.constructed import ConstructedPlacement

__version__ = "0.4.0"
__author__ = "Fan Zhang"


@dataclass
class MorphologyReport:
    """Typed morphology summary of an aggregate (spec 2.5, deviation N3).

    Attribute names are snake_case; `export_yaml` maps them to the legacy
    capitalized snapshot keys ("Rg", "Df_estimated", ...).
    """
    rg: float
    df_est: float
    r2: float
    r_centers: np.ndarray
    pair_correlation: np.ndarray
    com: np.ndarray
    n: int

def generate(
    n_particles: int,
    df: float,
    kf: float,
    method: Literal["pca", "cca"] = "pca",
    scaling: Literal["count", "mass"] = "mass",
    placement: Literal["sampled", "solved", "constructed"] = "solved",
    particle_dist: ParticleDistribution | None = None,
    overlap_tolerance: float = 1e-5,
    seed: int | None = None,
    **kwargs,
) -> Aggregate:
    """
    High-level API to generate a fractal aggregate.

    Args:
        n_particles (int): Target number of particles.
        df (float): Fractal dimension (typically 1.5 - 2.5).
        kf (float): Fractal prefactor (typically 1.0 - 2.0).
        method (str): Algorithm family: 'pca' (particle-cluster) or
            'cca' (cluster-cluster). 'fracval' is a deprecated alias for
            (cca, mass, constructed).
        scaling (str): 'count' (Filippov 2000 count weighting) or 'mass'
            (Moran 2019 mass weighting; polydispersity-correct). Default 'mass'.
        placement (str): 'sampled' (Monte Carlo), 'solved' (closed-form
            tangency; default), or 'constructed' (FracVAL contact
            construction; cca only).
        particle_dist: Particle radius distribution (defaults to Monodisperse(1.0)).
        overlap_tolerance (float): Allowed overlap between spheres.
        seed (int): Seed for reproducible generation (None = fresh entropy).

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
        scaling=scaling,
        placement=placement,
        particle_dist=particle_dist,
        overlap_tolerance=overlap_tolerance,
        seed=seed,
        **kwargs
    )

    return generator.generate()

def analyze(aggregate: Aggregate) -> MorphologyReport:
    """
    Compute the core morphological properties of an aggregate.
    """
    rg = radius_of_gyration(aggregate)
    r_centers, c_r = pair_correlation_function(aggregate)
    df_est, r2, _ = estimate_fractal_dimension(
        r_centers, c_r,
        r_min=float(np.mean(aggregate.radii)), r_max=rg,
    )
    return MorphologyReport(
        rg=rg,
        df_est=df_est,
        r2=r2,
        r_centers=r_centers,
        pair_correlation=c_r,
        com=center_of_mass(aggregate),
        n=aggregate.current_size,
    )


__all__ = [
    "generate",
    "analyze",
    "MorphologyReport",
    "Aggregate",
    "Monodisperse",
    "LognormalDistribution",
    "radius_of_gyration",
    "center_of_mass",
    "pair_correlation_function",
    "estimate_fractal_dimension",
    "plot_pair_correlation",
    "export_yaml",
    "export_render",
    "export_rotation_video",
    "export_vtm",
    "export_vtk",
    "PCAGenerator",
    "CCAGenerator",
    "SolvedPlacement",
    "SampledPlacement",
    "ConstructedPlacement",
]
