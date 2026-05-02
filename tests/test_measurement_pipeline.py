"""Тесты core/measurement_pipeline.py."""
from pathlib import Path

import numpy as np

from core.leaf_measure import build_fractions
from core.measurement_pipeline import (
    CropItem,
    PipelineSettings,
    measure_one,
    metrics_to_row,
    run_pipeline,
)


def _crop(h=80, w=40):
    """Простой «лист» --- белый прямоугольник на чёрном фоне."""
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :] = 255
    return out


def _settings(tmp_path, files=None, fmt="csv", n_sections=3, px_per_unit=None):
    return PipelineSettings(
        files=files or [],
        out_dir=tmp_path,
        n_sections=n_sections,
        unit_name="мм",
        px_per_unit=px_per_unit,
        export_format=fmt,
    )


class TestPipelineSettings:
    def test_no_scale_means_pixels(self, tmp_path):
        s = _settings(tmp_path, px_per_unit=None)
        assert s.unit == "px"
        assert s.scale == 1.0

    def test_scale_inverts_px_per_unit(self, tmp_path):
        s = _settings(tmp_path, px_per_unit=10.0)
        assert s.unit == "мм"
        assert abs(s.scale - 0.1) < 1e-9


class TestMeasureOne:
    def test_returns_metrics_and_writes_vis(self, tmp_path):
        vis_dir = tmp_path / "vis"
        vis_dir.mkdir()
        res = measure_one(
            CropItem(name="leaf01", crop=_crop()),
            build_fractions(3), "px", 1.0, vis_dir,
        )
        assert res.name == "leaf01"
        assert res.metrics.length > 0
        assert res.vis_path.exists()
        assert res.vis_path.stat().st_size > 0


class TestMetricsToRow:
    def _row(self, tmp_path, with_crop_col):
        vis_dir = tmp_path / "vis"
        vis_dir.mkdir()
        res = measure_one(
            CropItem(name="leaf01", crop=_crop()),
            build_fractions(3), "px", 1.0, vis_dir,
        )
        return res, metrics_to_row(res, build_fractions(3), "px", with_crop_col=with_crop_col)

    def test_no_crop_column_by_default(self, tmp_path):
        _, row = self._row(tmp_path, with_crop_col=False)
        assert "crop" not in row

    def test_crop_column_added_when_requested(self, tmp_path):
        res, row = self._row(tmp_path, with_crop_col=True)
        assert row["crop"] == str(res.vis_path)

    def test_image_first_then_length_and_width(self, tmp_path):
        _, row = self._row(tmp_path, with_crop_col=False)
        keys = list(row.keys())
        assert keys[0] == "image"
        assert "length_px" in keys
        assert "width_px" in keys
        assert keys.index("length_px") < keys.index("width_px")

    def test_section_columns_for_each_fraction(self, tmp_path):
        _, row = self._row(tmp_path, with_crop_col=False)
        for f in build_fractions(3):
            assert f"width_{f:.3f}_px" in row

    def test_two_decimals_format(self, tmp_path):
        _, row = self._row(tmp_path, with_crop_col=False)
        assert "." in row["length_px"]
        assert len(row["length_px"].split(".")[1]) == 2


class TestRunPipeline:
    def test_two_files_two_rows_csv(self, tmp_path):
        def source(path):
            yield CropItem(name=Path(path).stem, crop=_crop())

        files = ["a.png", "b.png"]
        result = run_pipeline(files, source, _settings(tmp_path, files=files))
        assert result.n_ok == 2
        assert len(result.thumb_paths) == 2
        assert (tmp_path / "measurements.csv").exists()
        assert (tmp_path / "vis").is_dir()
        assert len(list((tmp_path / "vis").glob("*.png"))) == 2

    def test_source_error_recorded_pipeline_continues(self, tmp_path):
        def source(path):
            if "bad" in path:
                raise RuntimeError("boom")
            yield CropItem(name=Path(path).stem, crop=_crop())

        files = ["good.png", "bad.png", "other.png"]
        result = run_pipeline(files, source, _settings(tmp_path, files=files))
        assert result.n_ok == 2
        assert len(result.errors) == 1
        assert "bad.png" in result.errors[0]

    def test_progress_callback_can_cancel(self, tmp_path):
        def source(path):
            yield CropItem(name=Path(path).stem, crop=_crop())

        files = ["a.png", "b.png", "c.png"]
        # Прерываем на 2-м файле (i=1).
        result = run_pipeline(
            files, source, _settings(tmp_path, files=files),
            on_progress=lambda i, n: i < 1,
        )
        assert result.n_ok == 1

    def test_xlsx_export(self, tmp_path):
        def source(path):
            yield CropItem(name=Path(path).stem, crop=_crop())

        files = ["x.png"]
        run_pipeline(files, source, _settings(tmp_path, files=files, fmt="xlsx"))
        assert (tmp_path / "measurements.xlsx").exists()
        assert not (tmp_path / "measurements.csv").exists()

    def test_no_results_no_export_file(self, tmp_path):
        def source(path):
            return iter([])   # источник ничего не выдаёт

        result = run_pipeline(["empty.png"], source, _settings(tmp_path, files=["empty.png"]))
        assert result.n_ok == 0
        assert not (tmp_path / "measurements.csv").exists()
        assert result.errors == []

    def test_source_yields_multiple_crops_per_file(self, tmp_path):
        def source(path):
            for i in range(3):
                yield CropItem(name=f"{Path(path).stem}_{i}", crop=_crop())

        result = run_pipeline(["x.png"], source, _settings(tmp_path, files=["x.png"]))
        assert result.n_ok == 3
