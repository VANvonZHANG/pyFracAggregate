import pytest
import pyFracAggregate as pfa


@pytest.mark.parametrize("method", ['pca', 'cca', 'fracval', 'tdcca'])
def test_all_methods_generate(method):
    n = 16 if method == 'tdcca' else 30
    agg = pfa.generate(n_particles=n, df=1.8, kf=1.3, method=method)
    assert agg.current_size == n
    assert agg.positions.shape == (n, 3)
    assert agg.radii.shape == (n,)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown generation method"):
        pfa.generate(n_particles=10, df=1.8, kf=1.3, method='nonexistent')


def test_removed_flage_pca_raises_helpful_error():
    with pytest.raises(ValueError, match="flage_pca.*has been removed"):
        pfa.generate(n_particles=10, df=1.8, kf=1.3, method='flage_pca')


def test_removed_flage_cca_raises_helpful_error():
    with pytest.raises(ValueError, match="flage_cca.*has been removed"):
        pfa.generate(n_particles=10, df=1.8, kf=1.3, method='flage_cca')


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_placement_param_forwarded():
    agg = pfa.generate(n_particles=20, df=1.8, kf=1.3, method='pca', placement='random')
    assert agg.current_size == 20
