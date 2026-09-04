import time
import pytest
import numpy as np
import pyFracAggregate as pfa

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.mark.benchmark
def test_pca_default_is_fast():
    n = 200
    df, kf = 1.8, 1.3
    np.random.seed(42)
    t0 = time.perf_counter()
    agg = pfa.generate(n_particles=n, df=df, kf=kf, method='pca')
    elapsed = time.perf_counter() - t0
    assert agg.current_size == n
    assert elapsed < 30.0


@pytest.mark.benchmark
def test_cca_default_is_fast():
    n = 100
    df, kf = 1.8, 1.3
    np.random.seed(42)
    t0 = time.perf_counter()
    agg = pfa.generate(n_particles=n, df=df, kf=kf, method='cca')
    elapsed = time.perf_counter() - t0
    assert agg.current_size == n
    assert elapsed < 30.0
