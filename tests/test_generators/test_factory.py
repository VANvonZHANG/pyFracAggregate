import pytest
import pyFracAggregate as pfa


@pytest.mark.parametrize("method", ['pca', 'cca'])
def test_all_methods_generate(method):
    agg = pfa.generate(n_particles=30, df=1.8, kf=1.3, method=method, seed=0)
    assert agg.current_size == 30
    assert agg.positions.shape == (30, 3)
    assert agg.radii.shape == (30,)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown generation method"):
        pfa.generate(n_particles=10, df=1.8, kf=1.3, method='nonexistent')


def test_removed_flage_pca_raises_helpful_error():
    with pytest.raises(ValueError, match="flage_pca.*has been removed"):
        pfa.generate(n_particles=10, df=1.8, kf=1.3, method='flage_pca')


def test_removed_flage_cca_raises_helpful_error():
    with pytest.raises(ValueError, match="flage_cca.*has been removed"):
        pfa.generate(n_particles=10, df=1.8, kf=1.3, method='flage_cca')


def test_placement_param_forwarded():
    agg = pfa.generate(n_particles=20, df=1.8, kf=1.3, method='pca',
                       placement='sampled', seed=0)
    assert agg.current_size == 20
