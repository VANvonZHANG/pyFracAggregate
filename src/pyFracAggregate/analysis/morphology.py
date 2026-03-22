import numpy as np
from pyFracAggregate.core.aggregate import Aggregate

def center_of_mass(aggregate: Aggregate) -> np.ndarray:
    """
    Calculate the center of mass of the aggregate.
    
    Args:
        aggregate (Aggregate): The aggregate object.
        
    Returns:
        np.ndarray: A 1D array of shape (3,) representing the (x, y, z) coordinates of the center of mass.
    """
    if aggregate.current_size == 0:
        return np.zeros(3)
        
    masses = aggregate.masses
    positions = aggregate.positions
    
    total_mass = np.sum(masses)
    if total_mass == 0:
        return np.mean(positions, axis=0)
        
    # sum(m_i * r_i) / sum(m_i)
    com = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass
    return com

def radius_of_gyration(aggregate: Aggregate) -> float:
    """
    Calculate the radius of gyration (Rg) of the aggregate.
    Uses the parallel axis theorem to account for the finite size of the primary particles.
    For a solid sphere, the radius of gyration about its own center is sqrt(3/5) * r.
    
    Args:
        aggregate (Aggregate): The aggregate object.
        
    Returns:
        float: The radius of gyration.
    """
    if aggregate.current_size == 0:
        return 0.0
        
    if aggregate.current_size == 1:
        # For a single sphere, Rg = sphere radius according to Filippov 2000 definition eq [4]
        # Or more accurately based on eq [4]: Rg^2 = a^2 for N=1. 
        # But for polydisperse spheres with parallel axis theorem: Rg^2 = 3/5 * r^2.
        # Wait, the Filippov paper uses: Rg^2 = 1/N sum((ri - r0)^2 + a^2).
        # Let's check the Morán 2019 FracVAL equation for polydisperse:
        # Rg^2 = 1/m_a sum(m_i * [(R_i - R_c)^2 + r_{g,i}^2]) where r_{g,i}^2 = 3/5 * r_{p,i}^2
        # Let's implement the standard physical Rg (with 3/5 factor).
        return np.sqrt(3.0 / 5.0) * aggregate.radii[0]

    masses = aggregate.masses
    positions = aggregate.positions
    radii = aggregate.radii
    
    total_mass = np.sum(masses)
    if total_mass == 0:
        return 0.0
        
    com = center_of_mass(aggregate)
    
    # Distance squared from CoM to each particle center
    dist_sq = np.sum((positions - com) ** 2, axis=1)
    
    # Intrinsic Rg squared of each solid sphere: 3/5 * r^2
    intrinsic_rg_sq = (3.0 / 5.0) * (radii ** 2)
    
    # sum(m_i * (dist_sq + intrinsic_rg_sq)) / M
    rg_sq = np.sum(masses * (dist_sq + intrinsic_rg_sq)) / total_mass
    
    return float(np.sqrt(rg_sq))
