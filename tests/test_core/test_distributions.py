import pytest
import numpy as np
from pyFracAggregate.core.distributions import Monodisperse, LognormalDistribution

def test_monodisperse():
    dist = Monodisperse(1.5)
    samples = dist.sample(100)
    assert len(samples) == 100
    assert np.all(samples == 1.5)
    
    with pytest.raises(ValueError, match="positive"):
        Monodisperse(-1)

def test_lognormal_distribution():
    mean_geo = 15.0
    std_geo = 1.5
    dist = LognormalDistribution(mean=mean_geo, std=std_geo)
    samples = dist.sample(100000)
    
    assert len(samples) == 100000
    assert np.all(samples > 0)
    
    # Calculate empirical log-mean and log-std
    log_samples = np.log(samples)
    emp_log_mean = np.mean(log_samples)
    emp_log_std = np.std(log_samples)
    
    expected_log_mean = np.log(mean_geo)
    expected_log_std = np.log(std_geo)
    
    # Check if they are close
    np.testing.assert_allclose(emp_log_mean, expected_log_mean, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(emp_log_std, expected_log_std, rtol=1e-2, atol=1e-2)

    with pytest.raises(ValueError, match="positive"):
        LognormalDistribution(-1, 1.5)
    with pytest.raises(ValueError, match="positive"):
        LognormalDistribution(15, -1)
        
def test_lognormal_distribution_std_one():
    dist = LognormalDistribution(mean=15.0, std=1.0)
    samples = dist.sample(10)
    assert np.allclose(samples, 15.0)
