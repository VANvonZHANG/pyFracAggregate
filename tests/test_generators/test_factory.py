import pytest
import pyFracAggregate as pfa


@pytest.mark.parametrize("method", ['pca', 'cca', 'fracval', 'flage_pca', 'flage_cca', 'tdcca'])
def test_all_methods_generate(method):
    """All registered methods should produce a valid aggregate."""
    n = 16 if method == 'tdcca' else 30
    agg = pfa.generate(n_particles=n, df=1.8, kf=1.3, method=method)
    assert agg.current_size == n
    assert agg.positions.shape == (n, 3)
    assert agg.radii.shape == (n,)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown generation method"):
        pfa.generate(n_particles=10, df=1.8, kf=1.3, method='nonexistent')
