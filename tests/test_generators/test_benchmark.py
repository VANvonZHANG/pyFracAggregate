import time
import pytest
import numpy as np
import pyFracAggregate as pfa


@pytest.mark.benchmark
def test_pca_flage_not_slower_than_pca():
    """FLAGE PCA should not be dramatically slower than Filippov PCA."""
    n = 200
    df, kf = 1.8, 1.3

    np.random.seed(42)
    t0 = time.perf_counter()
    agg1 = pfa.generate(n_particles=n, df=df, kf=kf, method='pca')
    t_filippov = time.perf_counter() - t0

    np.random.seed(42)
    t0 = time.perf_counter()
    agg2 = pfa.generate(n_particles=n, df=df, kf=kf, method='flage_pca')
    t_flage = time.perf_counter() - t0

    assert agg1.current_size == n
    assert agg2.current_size == n

    # Allow 2x + small constant for overhead at moderate N
    assert t_flage < t_filippov * 2.0 + 0.5


@pytest.mark.benchmark
def test_cca_flage_not_slower_than_cca():
    """FLAGE CCA should not be dramatically slower than Filippov CCA."""
    n = 100
    df, kf = 1.8, 1.3

    np.random.seed(42)
    t0 = time.perf_counter()
    agg1 = pfa.generate(n_particles=n, df=df, kf=kf, method='cca')
    t_filippov = time.perf_counter() - t0

    np.random.seed(42)
    t0 = time.perf_counter()
    agg2 = pfa.generate(n_particles=n, df=df, kf=kf, method='flage_cca')
    t_flage = time.perf_counter() - t0

    assert agg1.current_size == n
    assert agg2.current_size == n

    assert t_flage < t_filippov * 2.0 + 0.5
