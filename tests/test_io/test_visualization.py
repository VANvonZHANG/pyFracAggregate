import os
import sys

import numpy as np
import pytest

import pyFracAggregate as pfa
from pyFracAggregate.io import visualization as viz
from pyFracAggregate.io.visualization import (
    _build_plotter,
    _framing_distance,
    _orbit_camera_position,
    save_rotation_video,
    save_screenshot,
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
    def test_bounding_box_diagonal(self):
        assert _framing_distance((0.0, 2.0, 0.0, 4.0, 0.0, 6.0)) == pytest.approx(1.5 * np.sqrt(56.0))

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


class TestSaveScreenshot:
    def test_creates_png(self, small_aggregate, tmp_path):
        path = str(tmp_path / "render.png")
        save_screenshot(small_aggregate, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_custom_color_opacity(self, small_aggregate, tmp_path):
        path = str(tmp_path / "render_red.png")
        save_screenshot(small_aggregate, path, color="red", opacity=0.5)
        assert os.path.exists(path)

    def test_camera_presets(self, small_aggregate, tmp_path):
        for preset in ["iso", "xy", "xz", "yz"]:
            path = str(tmp_path / f"render_{preset}.png")
            save_screenshot(small_aggregate, path, camera_position=preset)
            assert os.path.exists(path)

    def test_color_by_radius_renders(self, small_aggregate, tmp_path):
        path = str(tmp_path / "render_radius.png")
        save_screenshot(small_aggregate, path, color_by="radius",
                        window_size=(320, 240))
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_rejects_non_png(self, small_aggregate, tmp_path):
        path = str(tmp_path / "render.jpg")
        with pytest.raises(ValueError, match="png"):
            save_screenshot(small_aggregate, path)

    def test_invalid_color_by_raises(self, small_aggregate, tmp_path):
        path = str(tmp_path / "render.png")
        with pytest.raises(ValueError, match="color_by"):
            save_screenshot(small_aggregate, path, color_by="mass")


class TestSaveRotationVideo:
    def test_creates_mp4(self, small_aggregate, tmp_path):
        path = str(tmp_path / "rotation.mp4")
        save_rotation_video(small_aggregate, path, n_frames=4, fps=4,
                            window_size=(320, 240))
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_rejects_non_mp4(self, small_aggregate, tmp_path):
        path = str(tmp_path / "rotation.gif")
        with pytest.raises(ValueError, match="mp4"):
            save_rotation_video(small_aggregate, path)

    def test_rejects_zero_frames(self, small_aggregate, tmp_path):
        path = str(tmp_path / "rotation.mp4")
        with pytest.raises(ValueError, match="n_frames"):
            save_rotation_video(small_aggregate, path, n_frames=0)

    def test_camera_actually_rotates(self, small_aggregate, tmp_path):
        """Regression: v0.4 export_rotation_video produced a static camera.

        Frames at 180 deg and 300 deg must differ from frame 0 by far more
        than encoding noise (measured noise floor of the bug: MAD ~ 0.01).
        """
        path = str(tmp_path / "rotation.mp4")
        save_rotation_video(small_aggregate, path, n_frames=6, fps=6,
                            window_size=(320, 240))
        import imageio.v2 as imageio

        with imageio.get_reader(path) as reader:
            f0 = np.asarray(reader.get_data(0), dtype=np.int16)
            f3 = np.asarray(reader.get_data(3), dtype=np.int16)
            f5 = np.asarray(reader.get_data(5), dtype=np.int16)
        assert np.abs(f0 - f3).mean() > 1.0
        assert np.abs(f0 - f5).mean() > 1.0

    def test_missing_imageio_raises(self, small_aggregate, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "imageio", None)
        path = str(tmp_path / "rotation.mp4")
        with pytest.raises(ImportError, match="imageio"):
            save_rotation_video(small_aggregate, path)

    def test_writer_creation_failure_closes_plotter(
        self, small_aggregate, tmp_path, monkeypatch
    ):
        import imageio.v2 as imageio

        def _boom(*args, **kwargs):
            raise RuntimeError("no ffmpeg plugin")

        monkeypatch.setattr(imageio, "get_writer", _boom)

        closed = []
        real_build = viz._build_plotter

        def spy_build(aggregate, **kwargs):
            plotter = real_build(aggregate, **kwargs)
            orig_close = plotter.close

            def _record_close():
                closed.append(True)
                orig_close()

            plotter.close = _record_close  # type: ignore[method-assign]
            return plotter

        monkeypatch.setattr(viz, "_build_plotter", spy_build)
        path = str(tmp_path / "rotation.mp4")
        with pytest.raises(RuntimeError, match="ffmpeg"):
            save_rotation_video(small_aggregate, path)
        assert closed == [True]


class TestPublicExports:
    def test_new_api_exported_old_api_removed(self):
        assert hasattr(pfa, "save_screenshot")
        assert hasattr(pfa, "save_rotation_video")
        assert "save_screenshot" in pfa.__all__
        assert "save_rotation_video" in pfa.__all__
        assert not hasattr(pfa, "export_render")
        assert not hasattr(pfa, "export_rotation_video")

    def test_version_bumped(self):
        assert pfa.__version__ == "0.6.1"
