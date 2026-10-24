"""Tests for the NISAR data-loading module.

Covers the velocity estimator and, with small synthetic HDF5 files in the
NISAR group layout, the GUNW/GOFF readers: reliability masking,
component-aware referencing, gap handling and GOFF SNR masking.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from glews.nisar import (
    NISARTimeseries,
    _compute_velocity,
    load_nisar_goff_stack,
    load_nisar_gunw_stack,
    read_goff_pairs,
)

GUNW_BASE = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH"
GOFF_BASE = "science/LSAR/GOFF/grids/frequencyA/pixelOffsets/HH"
DATES = ["20260101", "20260113", "20260125", "20260206"]


def _name(kind, ref, sec):
    return (f"NISAR_L2_PR_{kind}_001_098_A_016_001_4000_SH_{ref}T000000_{ref}T000030_"
            f"{sec}T000000_{sec}T000030_X05026_N_F_J_001.h5")


def _write_gunw(path, phase, cc, coh=None, iono=None, n=8):
    with h5py.File(path, "w") as f:
        g = f.create_group(GUNW_BASE)
        g["unwrappedPhase"] = phase.astype(np.float32)
        g["coherenceMagnitude"] = (np.full(phase.shape, 0.8) if coh is None else coh).astype(np.float32)
        g["connectedComponents"] = cc.astype(np.uint16)
        if iono is not None:
            g["ionospherePhaseScreen"] = iono.astype(np.float32)
        g["xCoordinates"] = 1000.0 + 80.0 * np.arange(phase.shape[1])
        g["yCoordinates"] = 5000.0 - 80.0 * np.arange(phase.shape[0])


def _write_goff(path, rg, az, snr):
    with h5py.File(path, "w") as f:
        for lay in ("layer1", "layer2", "layer3"):
            g = f.create_group(f"{GOFF_BASE}/{lay}")
            g.create_dataset("slantRangeOffset", data=rg.astype(np.float32)).attrs["units"] = b"meters"
            g.create_dataset("alongTrackOffset", data=az.astype(np.float32)).attrs["units"] = b"meters"
            g["snr"] = snr.astype(np.float32)
            g["xCoordinates"] = 1000.0 + 80.0 * np.arange(rg.shape[1])
            g["yCoordinates"] = 5000.0 - 80.0 * np.arange(rg.shape[0])


def _phase_for(disp_m):
    """Unwrapped phase giving LOS displacement disp_m (GLEWS sign convention)."""
    return -np.asarray(disp_m) * 4 * np.pi / 0.2384


class TestComputeVelocity:
    """Tests for the vectorized linear-velocity estimator."""

    def _make_dates(self, n=30, start=738886, step=12):
        return np.arange(start, start + n * step, step)

    def test_linear_displacement_recovers_velocity(self):
        """A pure linear ramp should recover the exact velocity."""
        dates = self._make_dates(30)
        t_yr = (dates - dates[0]).astype(float) / 365.25

        true_velocity = 0.06  # m/yr
        n_rows, n_cols = 5, 5
        displacement = np.zeros((len(dates), n_rows, n_cols))
        for i, t in enumerate(t_yr):
            displacement[i] = true_velocity * t

        velocity = _compute_velocity(dates, displacement)

        assert velocity.shape == (n_rows, n_cols)
        assert np.allclose(velocity, true_velocity, atol=1e-6)

    def test_negative_velocity(self):
        """Should correctly recover a negative (subsiding) velocity."""
        dates = self._make_dates(20)
        t_yr = (dates - dates[0]).astype(float) / 365.25

        true_velocity = -0.12
        displacement = np.zeros((len(dates), 3, 3))
        for i, t in enumerate(t_yr):
            displacement[i] = true_velocity * t

        velocity = _compute_velocity(dates, displacement)
        assert np.allclose(velocity, true_velocity, atol=1e-6)

    def test_insufficient_data_is_nan(self):
        """Pixels with fewer than 3 valid observations should be NaN."""
        dates = self._make_dates(5)
        displacement = np.full((5, 2, 2), np.nan)
        # Only 2 valid observations for every pixel — below the minimum of 3
        displacement[0] = 0.0
        displacement[1] = 0.01

        velocity = _compute_velocity(dates, displacement)
        assert np.all(np.isnan(velocity))

    def test_mixed_valid_and_nan_pixels(self):
        """Should compute velocity per-pixel independently, skipping
        pixels that don't have enough valid samples."""
        dates = self._make_dates(10)
        t_yr = (dates - dates[0]).astype(float) / 365.25

        displacement = np.zeros((len(dates), 2, 2))
        displacement[:, 0, 0] = 0.05 * t_yr  # fully valid, velocity 0.05
        displacement[:, 0, 1] = np.nan       # entirely missing
        displacement[:, 1, 0] = 0.05 * t_yr
        displacement[:2, 1, 0] = np.nan      # still enough valid points

        velocity = _compute_velocity(dates, displacement)

        assert np.isclose(velocity[0, 0], 0.05, atol=1e-6)
        assert np.isnan(velocity[0, 1])
        assert np.isclose(velocity[1, 0], 0.05, atol=1e-6)

    def test_noisy_data_approximate_recovery(self, rng):
        """With small noise, recovered velocity should be close to truth."""
        dates = self._make_dates(50)
        t_yr = (dates - dates[0]).astype(float) / 365.25

        true_velocity = 0.03
        displacement = np.zeros((len(dates), 4, 4))
        for i, t in enumerate(t_yr):
            displacement[i] = true_velocity * t
        displacement += rng.normal(0, 0.001, displacement.shape)

        velocity = _compute_velocity(dates, displacement)
        assert np.allclose(velocity, true_velocity, atol=0.01)


class TestNISARTimeseriesDataclass:
    """Sanity checks on the NISARTimeseries container itself."""

    def test_synthetic_fixture_shapes_are_consistent(self, synthetic_nisar_timeseries):
        ts = synthetic_nisar_timeseries
        assert isinstance(ts, NISARTimeseries)

        n_dates, n_rows, n_cols = ts.displacement.shape
        assert ts.dates.shape == (n_dates,)
        assert len(ts.date_strings) == n_dates
        assert ts.velocity.shape == (n_rows, n_cols)
        assert ts.coherence.shape == (n_rows, n_cols)
        assert ts.latitude.shape == (n_rows, n_cols)
        assert ts.longitude.shape == (n_rows, n_cols)
        assert ts.source == "GUNW"

    def test_synthetic_fixture_velocity_is_positive(self, synthetic_nisar_timeseries):
        # Fixture was built with a constant positive velocity trend.
        ts = synthetic_nisar_timeseries
        valid = np.isfinite(ts.velocity)
        assert valid.any()
        assert np.nanmean(ts.velocity) > 0


class TestGunwLoader:
    """Sequential 3-pair network on an 8x8 grid; the left two columns are
    the stable reference, pixel (4, 5) moves 1 cm per pair."""

    def _ref(self):
        m = np.zeros((8, 8), bool)
        m[:, :2] = True
        return m

    def _stack(self, tmp_path, offsets=(0.0, 0.0, 0.0), cc_fn=None, coh_fn=None, iono=None):
        for k in range(3):
            disp = np.zeros((8, 8))
            disp[4, 5] = 0.01
            phase = _phase_for(disp) + offsets[k]      # constant offset = unknown reference
            cc = np.ones((8, 8)) if cc_fn is None else cc_fn(k)
            coh = None if coh_fn is None else coh_fn(k)
            _write_gunw(tmp_path / _name("GUNW", DATES[k], DATES[k + 1]), phase, cc, coh, iono)
        return tmp_path

    def test_referencing_removes_per_pair_offset(self, tmp_path):
        d = self._stack(tmp_path, offsets=(3.0, -2.0, 7.5))
        ts = load_nisar_gunw_stack(d, reference_mask=self._ref())
        assert ts.metadata["network"] == "sequential"
        assert np.allclose(ts.displacement[:, 4, 5], [0, 0.01, 0.02, 0.03], atol=1e-6)
        assert np.allclose(ts.displacement[:, 0, 0], 0, atol=1e-6)

    def test_zero_phase_is_valid_not_fill(self, tmp_path):
        d = self._stack(tmp_path)
        ts = load_nisar_gunw_stack(d)
        assert np.isfinite(ts.displacement).all()

    def test_unreliable_pixels_break_the_chain(self, tmp_path):
        def cc_fn(k):
            cc = np.ones((8, 8))
            if k == 1:
                cc[4, 5] = 0                  # unwrapping unreliable in pair 2
            return cc
        d = self._stack(tmp_path, cc_fn=cc_fn)
        ts = load_nisar_gunw_stack(d, reference_mask=self._ref())
        assert np.isclose(ts.displacement[1, 4, 5], 0.01, atol=1e-6)
        assert np.isnan(ts.displacement[2:, 4, 5]).all()     # no zero-motion bridge
        assert ts.metadata["gap_policy"] == "break"

    def test_bridge_fills_gap_from_neighbouring_rates(self, tmp_path):
        def cc_fn(k):
            cc = np.ones((8, 8))
            if k == 1:
                cc[4, 5] = 0
            return cc
        d = self._stack(tmp_path, cc_fn=cc_fn)
        ts = load_nisar_gunw_stack(d, reference_mask=self._ref(), gap_policy="bridge")
        assert np.allclose(ts.displacement[:, 4, 5], [0, 0.01, 0.02, 0.03], atol=1e-6)
        assert ts.metadata["bridged_fraction"] > 0

    def test_unmasked_loader_keeps_unreliable_pixels(self, tmp_path):
        d = self._stack(tmp_path, cc_fn=lambda k: np.zeros((8, 8)))
        ts = load_nisar_gunw_stack(d, mask_unreliable=False)
        assert np.isfinite(ts.displacement).all()

    def test_other_component_is_dropped_when_referencing(self, tmp_path):
        def cc_fn(k):
            cc = np.ones((8, 8))
            cc[:, 6:] = 2                     # separate component on the right
            return cc
        d = self._stack(tmp_path, cc_fn=cc_fn)
        ts = load_nisar_gunw_stack(d, reference_mask=self._ref())
        assert np.isnan(ts.displacement[1:, :, 6:]).all()
        assert np.isfinite(ts.displacement[:, 4, 5]).all()
        assert ts.metadata["pairs"][0]["reference_component"] == 1

    def test_too_few_reference_pixels_masks_pair(self, tmp_path):
        def cc_fn(k):
            cc = np.ones((8, 8))
            if k == 0:
                cc[:, :2] = 0
            return cc
        d = self._stack(tmp_path, cc_fn=cc_fn)
        ts = load_nisar_gunw_stack(d, reference_mask=self._ref())
        assert np.isnan(ts.displacement[1:]).all()

    def test_ionosphere_correction(self, tmp_path):
        iono = np.full((8, 8), 0.5)
        iono[:, 4:] = 1.5                     # 1 rad differential screen
        d = self._stack(tmp_path, iono=None)
        for fp in d.glob("*.h5"):
            with h5py.File(fp, "a") as f:
                f[GUNW_BASE]["unwrappedPhase"][...] += iono
                f[GUNW_BASE]["ionospherePhaseScreen"] = iono.astype(np.float32)
        ref = self._ref()
        raw = load_nisar_gunw_stack(d, reference_mask=ref)
        cor = load_nisar_gunw_stack(d, reference_mask=ref, correct_ionosphere=True)
        assert not np.isclose(raw.displacement[3, 4, 5], 0.03, atol=1e-4)
        assert np.isclose(cor.displacement[3, 4, 5], 0.03, atol=1e-6)

    def test_bad_reference_shape_raises(self, tmp_path):
        d = self._stack(tmp_path)
        with pytest.raises(ValueError):
            load_nisar_gunw_stack(d, reference_mask=np.ones((3, 3), bool))

    def test_bad_gap_policy_raises(self, tmp_path):
        d = self._stack(tmp_path)
        with pytest.raises(ValueError):
            load_nisar_gunw_stack(d, gap_policy="zero")


class TestGoffLoader:
    def _stack(self, tmp_path, low_snr_pair=None):
        rng = np.random.default_rng(0)
        for k in range(3):
            rg = 0.3 + rng.normal(0, 0.05, (10, 10))      # 0.3 m common offset + noise
            az = -0.2 + rng.normal(0, 0.05, (10, 10))
            rg[5, 7] += 1.0                               # source moves 1 m range per pair
            az[5, 7] += 0.5
            snr = np.full((10, 10), 20.0)
            if low_snr_pair == k:
                snr[5, 7] = 2.0
            _write_goff(tmp_path / _name("GOFF", DATES[k], DATES[k + 1]), rg, az, snr)
        ref = np.zeros((10, 10), bool)
        ref[:, :3] = True
        return tmp_path, ref

    def test_pairs_are_referenced_with_noise_estimate(self, tmp_path):
        d, ref = self._stack(tmp_path)
        pairs = read_goff_pairs(d, layer="layer2", min_snr=5, reference_mask=ref)
        assert len(pairs) == 3 and pairs[0]["days"] == 12
        p = pairs[0]
        assert abs(np.median(p["range_offset"][ref])) < 1e-6
        assert np.isclose(p["range_offset"][5, 7], 1.0, atol=0.2)
        assert 0.02 < p["noise_sigma"][0] < 0.1

    def test_stack_has_both_components_and_snr_gap(self, tmp_path):
        d, ref = self._stack(tmp_path, low_snr_pair=1)
        ts = load_nisar_goff_stack(d, reference_mask=ref)
        assert ts.source == "GOFF" and ts.displacement_along_track is not None
        assert np.isclose(ts.displacement[1, 5, 7], 1.0, atol=0.2)
        assert np.isclose(ts.displacement_along_track[1, 5, 7], 0.5, atol=0.2)
        assert np.isnan(ts.displacement[2:, 5, 7]).all()
        assert ts.coherence[5, 7] == pytest.approx(2 / 3)

    def test_wrong_units_raise(self, tmp_path):
        d, ref = self._stack(tmp_path)
        for fp in d.glob("*.h5"):
            with h5py.File(fp, "a") as f:
                f[f"{GOFF_BASE}/layer2/slantRangeOffset"].attrs["units"] = b"pixels"
        assert read_goff_pairs(d) == []        # every file fails to read and is skipped

    def test_max_rate_masks_blunders_after_referencing(self, tmp_path):
        d, ref = self._stack(tmp_path)
        fp = sorted(d.glob("*.h5"))[1]
        with h5py.File(fp, "a") as f:
            f[f"{GOFF_BASE}/layer2/slantRangeOffset"][2, 8] = 40.0      # mismatched window
        kept = read_goff_pairs(d, reference_mask=ref)
        assert np.isfinite(kept[1]["range_offset"][2, 8])            # off by default
        pairs = read_goff_pairs(d, reference_mask=ref, max_rate=5.0 / 12)
        p = pairs[1]
        assert np.isnan(p["range_offset"][2, 8]) and np.isnan(p["along_track_offset"][2, 8])
        assert p["n_blunders"] == 1
        assert np.isfinite(p["range_offset"][5, 7])                  # 1.1 m / 12 d source survives
        assert pairs[0]["n_blunders"] == 0

    def test_max_rate_uses_both_components_and_pair_length(self, tmp_path):
        d, ref = self._stack(tmp_path)
        # source moves hypot(1.0, 0.5) ≈ 1.12 m in 12 days ≈ 0.093 m/day
        pairs = read_goff_pairs(d, reference_mask=ref, max_rate=0.08)
        assert all(np.isnan(p["range_offset"][5, 7]) for p in pairs)
        pairs = read_goff_pairs(d, reference_mask=ref, max_rate=0.11)
        assert all(np.isfinite(p["range_offset"][5, 7]) for p in pairs)

    def test_stack_passes_max_rate(self, tmp_path):
        d, ref = self._stack(tmp_path)
        ts = load_nisar_goff_stack(d, reference_mask=ref, max_rate=0.08)
        assert np.isnan(ts.displacement[1:, 5, 7]).all()
        assert ts.metadata["max_rate"] == 0.08
