"""Tests for native-grid GOFF cropping and track choice (glews.goff_crop)."""
import h5py
import numpy as np

from glews import goff_crop as gc
from glews.nisar import read_goff_pairs
from glews.tm import fwd

FID = "NISAR_L2_PR_GOFF_007_106_D_073_008_4000_SH_20261205T000000_20261205T000030_20261217T000000_20261217T000030_X05026_N_F_J_001"


def _product(path, epsg, lat, lon, lon0, n=400, post=80.0):
    cx, cy = fwd(lat, lon, lon0)
    x = cx - n / 2 * post + np.arange(n) * post + post / 2      # no node exactly on the centre
    y = cy + n / 2 * post - np.arange(n) * post - post / 2
    with h5py.File(path, "w") as f:
        f[f"{gc.BASE}/projection"] = np.int64(epsg)
        f[f"{gc.BASE}/xCoordinates"] = x
        f[f"{gc.BASE}/yCoordinates"] = y
        for lay in gc.LAYERS:
            for k in gc.FIELDS:
                v = np.add.outer(np.arange(n), np.arange(n)).astype("f4")
                f.create_dataset(f"{gc.BASE}/HH/{lay}/{k}", data=v if k != "snr" else np.full((n, n), 10, "f4"))
    return x, y


class TestCrop:
    def test_native_grid_and_crs_kept(self, tmp_path):
        src = tmp_path / "p.h5"
        x, y = _product(src, 32644, 30.38, 79.73, 81.0)
        with h5py.File(src, "r") as f:
            c = gc.crop_product(f, 30.38, 79.73)
        assert c["epsg"] == 32644
        assert c["layers"]["layer2"]["snr"].shape == (150, 150)       # 12 km at 80 m
        assert np.isin(c["x"], x).all() and np.isin(c["y"], y).all()  # no resampling
        out = tmp_path / "crop" / (FID + ".h5")
        out.parent.mkdir()
        gc.write_crop(out, c, FID)
        assert gc.crop_crs(out.parent) == "EPSG:32644"
        pairs = read_goff_pairs(out.parent, layer="layer2", min_snr=0)
        assert pairs[0]["sec_date"] == "20261217" and pairs[0]["days"] == 12

    def test_site_outside_grid(self, tmp_path):
        src = tmp_path / "p.h5"
        _product(src, 32645, 28.3, 85.5, 87.0)
        with h5py.File(src, "r") as f:
            try:
                gc.crop_product(f, 31.0, 85.5)
                assert False
            except ValueError:
                pass


def _res(path, frame, direction, sec, footprint):
    fid = f"NISAR_L2_PR_GOFF_001_{path:03d}_{direction[0]}_{frame:03d}_001_4000_SH_20261101T000000_20261101T000030_{sec}T000000_{sec}T000030_X_N_F_J_001"
    return dict(properties=dict(pathNumber=path, frameNumber=frame, flightDirection=direction, fileID=fid),
                footprint=footprint)


class TestChooseTrack:
    BIG = [(78.0, 29.0), (81.0, 29.0), (81.0, 32.0), (78.0, 32.0), (78.0, 29.0)]
    EDGE = [(79.70, 29.0), (81.0, 29.0), (81.0, 32.0), (79.70, 32.0), (79.70, 29.0)]   # cuts the crop

    def test_footprint_must_hold_crop(self):
        rs = [_res(106, 73, "DESCENDING", s, self.EDGE) for s in ("20261210", "20261222", "20270103")]
        assert gc.choose_track(rs, 30.38, 79.73, "20261201", "20270115") is None

    def test_most_baseline_pairs_then_ascending(self):
        rs = [_res(106, 73, "DESCENDING", s, self.BIG) for s in ("20261210", "20261222", "20270103")]
        rs += [_res(20, 25, "ASCENDING", s, self.BIG) for s in ("20261210", "20261222")]
        assert gc.choose_track(rs, 30.38, 79.73, "20261201", "20270115") == (106, 73, "DESCENDING")
        rs += [_res(20, 25, "ASCENDING", "20270110", self.BIG)]
        assert gc.choose_track(rs, 30.38, 79.73, "20261201", "20270115") == (20, 25, "ASCENDING")

    def test_duplicates_count_once(self):
        rs = [_res(106, 73, "DESCENDING", "20261210", self.BIG)] * 3
        rs += [_res(20, 25, "ASCENDING", s, self.BIG) for s in ("20261210", "20261222")]
        assert gc.choose_track(rs, 30.38, 79.73, "20261201", "20270115") == (20, 25, "ASCENDING")
