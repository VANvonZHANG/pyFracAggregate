from pyFracAggregate.generators.factory import get_generator
from pyFracAggregate.core.distributions import Monodisperse
import numpy as np

def test_density():
    # Parameters
    n = 1
    df = 2.0
    kf = 1.0
    radius = 10.0
    dist = Monodisperse(radius)
    
    # Test with default density (1.0)
    gen_default = get_generator('pca', n, df, kf, dist)
    agg_default = gen_default.generate()
    expected_mass_default = (4.0/3.0) * np.pi * (radius**3) * 1.0
    print(f"Default Density (1.0) - Mass: {agg_default.masses[0]:.4f}, Expected: {expected_mass_default:.4f}")
    
    # Test with custom density (2.5)
    gen_custom = get_generator('pca', n, df, kf, dist, density=2.5)
    agg_custom = gen_custom.generate()
    expected_mass_custom = (4.0/3.0) * np.pi * (radius**3) * 2.5
    print(f"Custom Density (2.5) - Mass: {agg_custom.masses[0]:.4f}, Expected: {expected_mass_custom:.4f}")
    
    if np.isclose(agg_custom.masses[0], expected_mass_custom):
        print("Test Passed: Density correctly affects mass calculation.")
    else:
        print("Test Failed: Density does not match expected mass.")

if __name__ == "__main__":
    test_density()
