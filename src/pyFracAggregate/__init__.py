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
    mass_pair_correlation_function,
    estimate_fractal_dimension,
    plot_pair_correlation
)
from pyFracAggregate.analysis.sandbox import (
    number_radius_function,
    mass_radius_function,
    number_sandbox_dimension,
    mass_sandbox_dimension,
    plot_sandbox
)
from pyFracAggregate.io.vtk import export_vtm, export_vtk
from pyFracAggregate.io.data import export_yaml
from pyFracAggregate.io.visualization import save_rotation_video, save_screenshot
from pyFracAggregate.generators.pca import PCAGenerator
from pyFracAggregate.generators.cca import CCAGenerator
from pyFracAggregate.generators.placement.solved import SolvedPlacement
from pyFracAggregate.generators.placement.sampled import SampledPlacement
from pyFracAggregate.generators.placement.constructed import ConstructedPlacement

__version__ = "0.6.0"
__author__ = "Fan Zhang"


@dataclass
class MorphologyReport:
    """Typed morphology summary of an aggregate, per measure.

    ``estimator`` records which family produced the report:
    "sandbox" (cumulative <N(r)>/<M(r)> curves, default) or "pcf"
    (differenced pair-correlation curves). Curve field contents follow
    the estimator; `export_yaml` serializes all fields under the v0.6
    snapshot key names.
    """
    rg: float
    com: np.ndarray
    n: int
    estimator: str
    df_num_est: float
    r2_num: float
    r_num: np.ndarray
    num_correlation: np.ndarray
    df_mass_est: float
    r2_mass: float
    r_mass: np.ndarray
    mass_correlation: np.ndarray

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

def analyze(
    aggregate: Aggregate,
    estimator: Literal["sandbox", "pcf"] = "sandbox",
) -> MorphologyReport:
    """
    Compute the core morphological properties of an aggregate.

    Args:
        aggregate (Aggregate): Cluster object.
        estimator (str): 'sandbox' (default) fits the cumulative
            mass-radius curves <N(r)> and <M(r)> — statistically robust for
            both measures; 'pcf' fits the differenced pair-correlation
            curves C(r) and C_m(r) — the counting path is the classic
            estimator, the mass path is noisy on single realizations.
    """
    if estimator not in ("sandbox", "pcf"):
        raise ValueError(
            f"estimator must be 'sandbox' or 'pcf', got {estimator!r}"
        )

    rg = radius_of_gyration(aggregate)
    com = center_of_mass(aggregate)

    if estimator == "sandbox":
        r_num, num_curve = number_radius_function(aggregate)
        df_num, r2_num, _ = number_sandbox_dimension(aggregate)
        r_mass, mass_curve = mass_radius_function(aggregate)
        df_mass, r2_mass, _ = mass_sandbox_dimension(aggregate)
    else:
        r_num, num_curve = pair_correlation_function(aggregate)
        r_mass, mass_curve = mass_pair_correlation_function(aggregate)
        r_min_fit = float(np.mean(aggregate.radii))
        df_num, r2_num, _ = estimate_fractal_dimension(
            r_num, num_curve, r_min=r_min_fit, r_max=rg
        )
        df_mass, r2_mass, _ = estimate_fractal_dimension(
            r_mass, mass_curve, r_min=r_min_fit, r_max=rg
        )

    return MorphologyReport(
        rg=rg,
        com=com,
        n=aggregate.current_size,
        estimator=estimator,
        df_num_est=df_num,
        r2_num=r2_num,
        r_num=r_num,
        num_correlation=num_curve,
        df_mass_est=df_mass,
        r2_mass=r2_mass,
        r_mass=r_mass,
        mass_correlation=mass_curve,
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
    "mass_pair_correlation_function",
    "number_radius_function",
    "mass_radius_function",
    "number_sandbox_dimension",
    "mass_sandbox_dimension",
    "plot_sandbox",
    "export_yaml",
    "save_screenshot",
    "save_rotation_video",
    "export_vtm",
    "export_vtk",
    "PCAGenerator",
    "CCAGenerator",
    "SolvedPlacement",
    "SampledPlacement",
    "ConstructedPlacement",
]
