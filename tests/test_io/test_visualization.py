import sys

import numpy as np
import pytest

import pyFracAggregate as pfa
from pyFracAggregate.io.visualization import (
    _build_plotter,
    _framing_distance,
    _orbit_camera_position,
)


@pytest.fixture
def small_aggregate():
    return pfa.generate(n_particles=20, df=1.8, kf=1.2, method="pca")


class TestOrbitCameraPosition:
    def test_azimuth_0(self):
        cam = _orbit_camera_position(np.zeros(3), 10.0, 0.0, 0.0)
        assert cam[0] == [10.0, 0.0, 0.0]

    def test_azimuth_90(self):
        cam = _orbit_camera_position(np.zeros(3), 10.0, 90.0, 0.0)
        assert np.allclose(cam[0], [0.0, 10.0, 0.0])

    def test_azimuth_180(self):
        cam = _orbit_camera_position(np.zeros(3), 10.0, 180.0, 0.0)
        assert np.allclose(cam[0], [-10.0, 0.0, 0.0])

    def test_elevation_90(self):
        cam = _orbit_camera_position(np.zeros(3), 10.0, 0.0, 90.0)
        assert np.allclose(cam[0], [0.0, 0.0, 10.0])

    def test_center_offset(self):
        cam = _orbit_camera_position(np.array([1.0, 2.0, 3.0]), 10.0, 0.0, 0.0)
        assert np.allclose(cam[0], [11.0, 2.0, 3.0])

    def test_structure_pos_focal_up(self):
        cam = _orbit_camera_position(np.array([1.0, 1.0, 1.0]), 5.0, 30.0, 30.0)
        assert len(cam) == 3
        assert cam[1] == [1.0, 1.0, 1.0]
        assert cam[2] == [0.0, 0.0, 1.0]


class TestFramingDistance:
    def test_largest_extent(self):
        assert _framing_distance((0.0, 2.0, 0.0, 4.0, 0.0, 6.0)) == 9.0

    def test_degenerate_bounds_fallback(self):
        assert _framing_distance((0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) == 1.5


class TestBuildPlotter:
    def test_solid_color_mesh_blocks_match_particles(self, small_aggregate):
        plotter = _build_plotter(
            small_aggregate, color="lightblue", opacity=0.8, color_by=None,
            cmap="viridis", background="white", window_size=(320, 240),
        )
        try:
            assert len(plotter.meshes) == 1
            assert len(plotter.meshes[0]) == small_aggregate.current_size
        finally:
            plotter.close()

    def test_color_by_radius_attaches_scalar_data(self, small_aggregate):
        plotter = _build_plotter(
            small_aggregate, color="lightblue", opacity=0.8, color_by="radius",
            cmap="viridis", background="white", window_size=(320, 240),
        )
        try:
            assert len(plotter.meshes) == 1
            data = plotter.meshes[0]
            assert "radius" in data.cell_data
            radii = np.unique(data.cell_data["radius"])
            assert np.all(np.isin(radii, small_aggregate.radii))
        finally:
            plotter.close()

    def test_empty_aggregate_raises(self):
        empty = pfa.Aggregate(max_particles=5)
        with pytest.raises(ValueError, match="empty"):
            _build_plotter(
                empty, color="red", opacity=1.0, color_by=None,
                cmap="viridis", background="white", window_size=(64, 48),
            )

    def test_invalid_color_by_raises(self, small_aggregate):
        with pytest.raises(ValueError, match="color_by"):
            _build_plotter(
                small_aggregate, color="red", opacity=1.0, color_by="mass",
                cmap="viridis", background="white", window_size=(64, 48),
            )
