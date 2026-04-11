import os
import pytest
import numpy as np
import pyFracAggregate as pfa


@pytest.fixture
def small_aggregate():
    return pfa.generate(n_particles=20, df=1.8, kf=1.2, method="pca")


class TestExportRender:
    def test_export_render_creates_png(self, small_aggregate, tmp_path):
        path = str(tmp_path / "render.png")
        pfa.export_render(small_aggregate, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_export_render_custom_color(self, small_aggregate, tmp_path):
        path = str(tmp_path / "render_red.png")
        pfa.export_render(small_aggregate, path, color="red", opacity=0.5)
        assert os.path.exists(path)

    def test_export_render_camera_presets(self, small_aggregate, tmp_path):
        for preset in ["iso", "xy", "xz", "yz"]:
            path = str(tmp_path / f"render_{preset}.png")
            pfa.export_render(small_aggregate, path, camera_position=preset)
            assert os.path.exists(path)
