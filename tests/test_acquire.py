"""Tests for the data acquisition module."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gews.acquire import (
    SceneInfo,
    _create_session,
    _extract_file_size_mb,
    _filename_from_url,
    _resolve_dataset,
    _resolve_platform,
    _resolve_product_type,
    download_scenes,
    search_scenes,
    select_track,
    summarize_scenes,
)


# ---------- helpers ----------

def _make_scene(
    granule: str = "S1-GUNW-test",
    start_time: str = "2026-06-01T00:00:00Z",
    path_number: int = 55,
    frame_number: int = 70,
    url: str = "https://example.com/test.nc",
    file_size_mb: float = 129.0,
    product_type: str = "GUNW",
) -> SceneInfo:
    return SceneInfo(
        granule=granule,
        start_time=start_time,
        path_number=path_number,
        frame_number=frame_number,
        polarization="VV",
        url=url,
        file_size_mb=file_size_mb,
        geometry={"type": "Point", "coordinates": [74.55, 36.42]},
        product_type=product_type,
    )


def _make_asf_product(
    file_id: str = "test-product",
    start_time: str = "2026-06-01T00:00:00Z",
    path_number: int = 55,
    frame_number: int = 70,
    bytes_val=129_000_000,
    url: str = "https://example.com/test.nc",
) -> MagicMock:
    product = MagicMock()
    product.properties = {
        "fileID": file_id,
        "startTime": start_time,
        "stopTime": "2026-06-12T00:00:00Z",
        "pathNumber": path_number,
        "frameNumber": frame_number,
        "polarization": "VV",
        "url": url,
        "bytes": bytes_val,
        "fileName": f"{file_id}.h5",
        "perpendicularBaseline": 42.5,
    }
    product.geometry = {"type": "Polygon", "coordinates": [[[74, 36], [75, 36], [75, 37], [74, 37], [74, 36]]]}
    return product


# ---------- _extract_file_size_mb ----------

class TestExtractFileSize:
    def test_integer_bytes(self):
        assert _extract_file_size_mb(129_000_000) == pytest.approx(129.0)

    def test_float_bytes(self):
        assert _extract_file_size_mb(129_000_000.0) == pytest.approx(129.0)

    def test_nisar_bytes_dict(self):
        bytes_dict = {
            "product.h5": {"bytes": 2_273_312_768, "format": "HDF5"},
            "product.xml": {"bytes": 6_000_000, "format": "XML"},
        }
        assert _extract_file_size_mb(bytes_dict) == pytest.approx(2279.3, rel=0.01)

    def test_zero(self):
        assert _extract_file_size_mb(0) == 0.0

    def test_none(self):
        assert _extract_file_size_mb(None) == 0.0


# ---------- _filename_from_url ----------

class TestFilenameFromUrl:
    def test_simple_url(self):
        assert _filename_from_url("https://example.com/data/product.h5") == "product.h5"

    def test_nc_file(self):
        assert _filename_from_url("https://example.com/S1-GUNW-test.nc") == "S1-GUNW-test.nc"

    def test_nested_path(self):
        result = _filename_from_url(
            "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/V1/NISAR_L2_PR_GUNW_030.h5"
        )
        assert result == "NISAR_L2_PR_GUNW_030.h5"


# ---------- resolve helpers ----------

class TestResolvers:
    def test_resolve_platform_nisar(self):
        import asf_search as asf
        assert _resolve_platform("NISAR") == asf.PLATFORM.NISAR

    def test_resolve_platform_sentinel(self):
        import asf_search as asf
        assert _resolve_platform("SENTINEL1") == asf.PLATFORM.SENTINEL1
        assert _resolve_platform("SENTINEL-1") == asf.PLATFORM.SENTINEL1

    def test_resolve_platform_unknown_defaults_sentinel(self):
        import asf_search as asf
        assert _resolve_platform("UNKNOWN") == asf.PLATFORM.SENTINEL1

    def test_resolve_product_type_gunw(self):
        import asf_search as asf
        assert _resolve_product_type("GUNW") == asf.PRODUCT_TYPE.GUNW

    def test_resolve_product_type_goff(self):
        import asf_search as asf
        assert _resolve_product_type("GOFF") == asf.PRODUCT_TYPE.GOFF

    def test_resolve_product_type_slc_default(self):
        import asf_search as asf
        assert _resolve_product_type("SLC") == asf.PRODUCT_TYPE.SLC

    def test_resolve_dataset_nisar(self):
        import asf_search as asf
        assert _resolve_dataset("NISAR", "GUNW") == asf.DATASET.NISAR

    def test_resolve_dataset_aria_s1_gunw(self):
        import asf_search as asf
        assert _resolve_dataset("SENTINEL1", "GUNW") == asf.DATASET.ARIA_S1_GUNW

    def test_resolve_dataset_sentinel_slc_returns_none(self):
        assert _resolve_dataset("SENTINEL1", "SLC") is None


# ---------- SceneInfo.from_asf_result ----------

class TestSceneInfoFromAsf:
    def test_basic_slc(self):
        product = _make_asf_product()
        scene = SceneInfo.from_asf_result(product, product_type="SLC")
        assert scene.product_type == "SLC"
        assert scene.file_size_mb == pytest.approx(129.0)

    def test_nisar_bytes_dict(self):
        product = _make_asf_product(
            bytes_val={"file.h5": {"bytes": 2_000_000_000, "format": "HDF5"}}
        )
        scene = SceneInfo.from_asf_result(product, product_type="GUNW")
        assert scene.product_type == "GUNW"
        assert scene.file_size_mb == pytest.approx(2000.0)

    def test_stop_time_captured(self):
        product = _make_asf_product()
        scene = SceneInfo.from_asf_result(product, product_type="GUNW")
        assert scene.stop_time == "2026-06-12T00:00:00Z"

    def test_perpendicular_baseline(self):
        product = _make_asf_product()
        scene = SceneInfo.from_asf_result(product)
        assert scene.perpendicular_baseline == 42.5


# ---------- search_scenes ----------

class TestSearchScenes:
    @patch("gews.acquire.asf")
    def test_slc_default(self, mock_asf):
        import asf_search
        mock_asf.PLATFORM = asf_search.PLATFORM
        mock_asf.PRODUCT_TYPE = asf_search.PRODUCT_TYPE
        mock_asf.DATASET = asf_search.DATASET
        mock_asf.BEAMMODE = asf_search.BEAMMODE
        mock_asf.search.return_value = [_make_asf_product()]

        config = {
            "site": {"name": "Test", "latitude": 30.0, "longitude": 80.0},
        }
        scenes = search_scenes(config)
        assert len(scenes) == 1
        assert scenes[0].product_type == "SLC"

        call_kwargs = mock_asf.search.call_args[1]
        assert call_kwargs["processingLevel"] == asf_search.PRODUCT_TYPE.SLC

    @patch("gews.acquire.asf")
    def test_nisar_gunw_search(self, mock_asf):
        import asf_search
        mock_asf.PLATFORM = asf_search.PLATFORM
        mock_asf.PRODUCT_TYPE = asf_search.PRODUCT_TYPE
        mock_asf.DATASET = asf_search.DATASET
        mock_asf.BEAMMODE = asf_search.BEAMMODE
        mock_asf.search.return_value = [
            _make_asf_product(file_id="NISAR_GUNW_1"),
            _make_asf_product(file_id="NISAR_GUNW_2", start_time="2026-06-12T00:00:00Z"),
        ]

        config = {
            "site": {"name": "Test", "latitude": 30.0, "longitude": 80.0},
            "acquire": {"platform": "NISAR", "product_types": ["GUNW"]},
        }
        scenes = search_scenes(config)
        assert len(scenes) == 2
        assert all(s.product_type == "GUNW" for s in scenes)

        call_kwargs = mock_asf.search.call_args[1]
        assert call_kwargs["dataset"] == asf_search.DATASET.NISAR
        assert call_kwargs["processingLevel"] == asf_search.PRODUCT_TYPE.GUNW
        assert "platform" not in call_kwargs

    @patch("gews.acquire.asf")
    def test_multiple_product_types(self, mock_asf):
        import asf_search
        mock_asf.PLATFORM = asf_search.PLATFORM
        mock_asf.PRODUCT_TYPE = asf_search.PRODUCT_TYPE
        mock_asf.DATASET = asf_search.DATASET
        mock_asf.BEAMMODE = asf_search.BEAMMODE
        mock_asf.search.side_effect = [
            [_make_asf_product(file_id="GUNW_1")],
            [_make_asf_product(file_id="GOFF_1")],
        ]

        config = {
            "site": {"name": "Test", "latitude": 30.0, "longitude": 80.0},
            "acquire": {"platform": "NISAR", "product_types": ["GUNW", "GOFF"]},
        }
        scenes = search_scenes(config)
        assert len(scenes) == 2
        assert mock_asf.search.call_count == 2


# ---------- select_track ----------

class TestSelectTrack:
    def test_auto_selects_most_populated(self):
        scenes = [
            _make_scene(path_number=55, start_time="2026-06-01T00:00:00Z"),
            _make_scene(path_number=55, start_time="2026-06-13T00:00:00Z"),
            _make_scene(path_number=63, start_time="2026-06-02T00:00:00Z"),
        ]
        selected = select_track(scenes)
        assert len(selected) == 2
        assert all(s.path_number == 55 for s in selected)

    def test_explicit_track(self):
        scenes = [
            _make_scene(path_number=55),
            _make_scene(path_number=63),
        ]
        selected = select_track(scenes, path_number=63)
        assert len(selected) == 1
        assert selected[0].path_number == 63

    def test_missing_track_raises(self):
        scenes = [_make_scene(path_number=55)]
        with pytest.raises(ValueError, match="Track 99 not found"):
            select_track(scenes, path_number=99)


# ---------- download_scenes ----------

class TestDownloadScenes:
    def test_skips_existing_h5(self, tmp_path):
        (tmp_path / "NISAR_GUNW_001.h5").write_text("exists")
        scenes = [
            _make_scene(
                url="https://example.com/NISAR_GUNW_001.h5",
                granule="NISAR_GUNW_001",
            ),
            _make_scene(
                url="https://example.com/NISAR_GUNW_002.h5",
                granule="NISAR_GUNW_002",
            ),
        ]

        with patch("gews.acquire.asf") as mock_asf:
            mock_asf.ASFSession.return_value = MagicMock()
            download_scenes(scenes, output_dir=tmp_path)

            urls_downloaded = mock_asf.download_urls.call_args[1]["urls"]
            assert len(urls_downloaded) == 1
            assert "NISAR_GUNW_002" in urls_downloaded[0]

    def test_skips_existing_nc(self, tmp_path):
        (tmp_path / "S1-GUNW-test.nc").write_text("exists")
        scenes = [
            _make_scene(url="https://example.com/S1-GUNW-test.nc"),
        ]

        with patch("gews.acquire.asf") as mock_asf:
            mock_asf.ASFSession.return_value = MagicMock()
            download_scenes(scenes, output_dir=tmp_path)
            mock_asf.download_urls.assert_not_called()


# ---------- _create_session ----------

class TestCreateSession:
    @patch.dict("os.environ", {}, clear=True)
    def test_netrc_fallback(self):
        with patch("gews.acquire.asf") as mock_asf:
            mock_session = MagicMock()
            mock_asf.ASFSession.return_value = mock_session
            session = _create_session()
            mock_session.auth_with_creds.assert_not_called()

    @patch.dict("os.environ", {"EARTHDATA_USER": "user", "EARTHDATA_PASS": "pass"})
    def test_env_vars(self):
        with patch("gews.acquire.asf") as mock_asf:
            mock_session = MagicMock()
            mock_asf.ASFSession.return_value = mock_session
            session = _create_session()
            mock_session.auth_with_creds.assert_called_once_with("user", "pass")

    def test_explicit_creds(self):
        with patch("gews.acquire.asf") as mock_asf:
            mock_session = MagicMock()
            mock_asf.ASFSession.return_value = mock_session
            session = _create_session(username="u", password="p")
            mock_session.auth_with_creds.assert_called_once_with("u", "p")


# ---------- summarize_scenes ----------

class TestSummarizeScenes:
    def test_empty(self):
        assert summarize_scenes([]) == "No scenes found."

    def test_mixed_product_types(self):
        scenes = [
            _make_scene(product_type="GUNW", start_time="2026-06-01T00:00:00Z"),
            _make_scene(product_type="GOFF", start_time="2026-06-12T00:00:00Z"),
        ]
        summary = summarize_scenes(scenes)
        assert "GOFF/GUNW" in summary
        assert "2 GOFF/GUNW products" in summary or "1 GOFF" in summary

    def test_zero_gap_filtered(self):
        scenes = [
            _make_scene(start_time="2026-06-01T00:00:00Z", path_number=55),
            _make_scene(start_time="2026-06-01T00:00:00Z", path_number=55, product_type="GOFF"),
            _make_scene(start_time="2026-06-13T00:00:00Z", path_number=55),
        ]
        summary = summarize_scenes(scenes)
        assert "median 12d" in summary
