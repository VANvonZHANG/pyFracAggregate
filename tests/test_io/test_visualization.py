import sys

import numpy as np
import pytest

import pyFracAggregate as pfa
from pyFracAggregate.io.visualization import (
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
