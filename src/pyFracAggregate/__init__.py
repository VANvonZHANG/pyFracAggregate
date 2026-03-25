from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.generators.factory import get_generator
from pyFracAggregate.core.distributions import Monodisperse, LognormalDistribution
from pyFracAggregate.analysis.morphology import radius_of_gyration, center_of_mass
from pyFracAggregate.analysis.correlation import pair_correlation_function
from pyFracAggregate.io.mesh import export_glb
from pyFracAggregate.io.vtk import export_vtm, export_vtk
from pyFracAggregate.io.data import export_to_json

__version__ = "0.1.0"
__author__ = "Fan Zhang"

def generate(
    n_particles: int,
    df: float,
    kf: float,
    method: str = 'pca',
    particle_dist = None,
    overlap_tolerance: float = 1e-5,
    **kwargs
) -> Aggregate:
    """
    High-level API to generate a fractal aggregate.
    
    Args:
        n_particles (int): Target number of particles.
        df (float): Fractal dimension (typically 1.5 - 2.5).
        kf (float): Fractal prefactor (typically 1.0 - 2.0).
        method (str): Algorithm to use ('pca', 'cca', 'fracval').
        particle_dist: Particle radius distribution (defaults to Monodisperse(1.0)).
        overlap_tolerance (float): Allowed overlap between spheres.
        
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
        **kwargs
    )
    
    return generator.generate()

def analyze(aggregate: Aggregate):
    """
    Compute core morphological properties.
    """
    return {
        "Rg": radius_of_gyration(aggregate),
        "CoM": center_of_mass(aggregate),
        "N": aggregate.current_size
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
    "export_glb",
    "export_vtm",
    "export_vtk",
    "export_to_json"
]
