import time
import pytest
import pyFracAggregate as pfa



@pytest.mark.benchmark
def test_pca_default_is_fast():
    n = 200
    df, kf = 1.8, 1.3
    t0 = time.perf_counter()
    agg = pfa.generate(n_particles=n, df=df, kf=kf, method='pca', seed=42)
    elapsed = time.perf_counter() - t0
    assert agg.current_size == n
    assert elapsed < 30.0


@pytest.mark.benchmark
def test_cca_default_is_fast():
    n = 100
    df, kf = 1.8, 1.3
    t0 = time.perf_counter()
    agg = pfa.generate(n_particles=n, df=df, kf=kf, method='cca', seed=42)
    elapsed = time.perf_counter() - t0
    assert agg.current_size == n
    assert elapsed < 30.0
