"""Tests for the transfer learning classifier module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from glews.classifier import (
    FEATURE_NAMES,
    LandslideTrainingData,
    PrecursorClassifier,
    PrecursorFeatureExtractor,
    _sigmoid,
    train_precursor_model,
)


# ------------------------------------------------------------------ #
#  Feature extraction tests                                           #
# ------------------------------------------------------------------ #

class TestPrecursorFeatureExtractor:
    """Tests for PrecursorFeatureExtractor."""

    def test_feature_vector_length(self):
        """Feature vector length must match FEATURE_NAMES."""
        extractor = PrecursorFeatureExtractor()
        dates = np.arange(50) * 12.0 + 738000
        disp = np.cumsum(np.random.default_rng(0).normal(0, 0.01, 50))
        feats = extractor.extract_features(dates, disp)
        assert len(feats) == len(FEATURE_NAMES)

    def test_features_are_finite(self):
        """All features must be finite (no NaN/Inf)."""
        extractor = PrecursorFeatureExtractor()
        dates = np.arange(30) * 12.0 + 738000
        # Constant displacement — zero variance edge case
        disp = np.ones(30) * 5.0
        feats = extractor.extract_features(dates, disp)
        assert np.all(np.isfinite(feats))

    def test_features_finite_with_nans(self):
        """Features should handle NaN values in input."""
        extractor = PrecursorFeatureExtractor()
        dates = np.arange(30) * 12.0 + 738000
        disp = np.cumsum(np.random.default_rng(1).normal(0, 0.01, 30))
        disp[5] = np.nan
        disp[10] = np.nan
        feats = extractor.extract_features(dates, disp)
        assert np.all(np.isfinite(feats))
        assert len(feats) == len(FEATURE_NAMES)

    def test_too_short_series_returns_zeros(self):
        """Series shorter than 3 points returns zero vector."""
        extractor = PrecursorFeatureExtractor()
        dates = np.array([738000.0, 738012.0])
        disp = np.array([0.0, 0.01])
        feats = extractor.extract_features(dates, disp)
        assert np.all(feats == 0)

    def test_exponential_series_has_high_asymmetry(self):
        """An exponentially accelerating series should have high temporal asymmetry."""
        extractor = PrecursorFeatureExtractor()
        dates = np.arange(60) * 12.0 + 738000
        t_norm = np.linspace(0, 1, 60)
        disp = 0.5 * (np.exp(5 * t_norm) - 1) / np.exp(5)
        feats = extractor.extract_features(dates, disp)
        # temporal_asymmetry is feature index 10
        assert feats[10] > 0.3  # last 30% holds more than 30% of acceleration

    def test_velocity_ratio_for_acceleration(self):
        """Accelerating series should have velocity ratio > 1."""
        extractor = PrecursorFeatureExtractor()
        dates = np.arange(60) * 12.0 + 738000
        t_norm = np.linspace(0, 1, 60)
        # Strong acceleration in second half
        disp = 0.01 * t_norm + 0.5 * (np.exp(4 * t_norm) - 1) / np.exp(4)
        feats = extractor.extract_features(dates, disp)
        # velocity_ratio is feature index 8
        assert feats[8] > 1.0


# ------------------------------------------------------------------ #
#  Logistic regression tests                                          #
# ------------------------------------------------------------------ #

class TestPrecursorClassifier:
    """Tests for PrecursorClassifier."""

    def test_fit_returns_losses(self):
        """fit() should return a list of losses matching n_iter."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 5))
        y = (X[:, 0] > 0).astype(float)
        model = PrecursorClassifier()
        losses = model.fit(X, y, n_iter=50)
        assert len(losses) == 50

    def test_loss_decreases(self):
        """Loss should generally decrease during training."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 5))
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        model = PrecursorClassifier()
        losses = model.fit(X, y, lr=0.1, n_iter=200)
        # Final loss should be less than initial loss
        assert losses[-1] < losses[0]
        # And the trend should be downward (check in blocks)
        for i in range(0, len(losses) - 20, 20):
            block_start = np.mean(losses[i:i+10])
            block_end = np.mean(losses[i+10:i+20])
            # Allow slight non-monotonicity but overall trend
            assert block_end < block_start + 0.05

    def test_linearly_separable_perfect_accuracy(self):
        """On a linearly separable problem, should reach ~100% accuracy."""
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(0, 1, (n, 3))
        # Clear linear separation with margin
        y = (X[:, 0] + X[:, 1] + X[:, 2] > 1.0).astype(float)
        # Ensure both classes present
        if y.sum() < 5 or (1 - y).sum() < 5:
            pytest.skip("Degenerate random draw")

        model = PrecursorClassifier()
        model.fit(X, y, lr=0.5, n_iter=500)
        preds = model.predict(X)
        accuracy = np.mean(preds == y)
        assert accuracy >= 0.95

    def test_predict_proba_range(self):
        """Probabilities must be in [0, 1]."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 4))
        y = (X[:, 0] > 0).astype(float)
        model = PrecursorClassifier()
        model.fit(X, y, n_iter=50)
        proba = model.predict_proba(X)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_predict_single_sample(self):
        """predict_proba should accept a single 1D feature vector."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 4))
        y = (X[:, 0] > 0).astype(float)
        model = PrecursorClassifier()
        model.fit(X, y, n_iter=50)
        proba = model.predict_proba(X[0])
        assert proba.shape == (1,)

    def test_predict_before_fit_raises(self):
        """predict_proba before fit should raise RuntimeError."""
        model = PrecursorClassifier()
        with pytest.raises(RuntimeError):
            model.predict_proba(np.array([[1.0, 2.0]]))

    def test_save_before_fit_raises(self):
        """save before fit should raise RuntimeError."""
        model = PrecursorClassifier()
        with pytest.raises(RuntimeError):
            model.save("/tmp/should_not_exist.json")


# ------------------------------------------------------------------ #
#  Save / load round-trip tests                                       #
# ------------------------------------------------------------------ #

class TestSaveLoad:
    """Tests for model serialization."""

    def test_save_load_round_trip(self, tmp_path):
        """Loaded model should produce identical predictions."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 5))
        y = (X[:, 0] > 0).astype(float)

        model = PrecursorClassifier()
        model.fit(X, y, n_iter=100)

        path = tmp_path / "model.json"
        model.save(path)

        loaded = PrecursorClassifier.load(path)
        proba_orig = model.predict_proba(X)
        proba_loaded = loaded.predict_proba(X)

        np.testing.assert_allclose(proba_orig, proba_loaded, rtol=1e-10)

    def test_save_produces_valid_json(self, tmp_path):
        """Saved model file should be valid JSON with expected keys."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 3))
        y = (X[:, 0] > 0).astype(float)

        model = PrecursorClassifier()
        model.fit(X, y, n_iter=10)

        path = tmp_path / "model.json"
        model.save(path)

        data = json.loads(path.read_text())
        assert "version" in data
        assert "feature_names" in data
        assert "weights" in data
        assert "bias" in data
        assert "mean" in data
        assert "std" in data
        assert len(data["weights"]) == 3

    def test_load_nonexistent_raises(self, tmp_path):
        """Loading a nonexistent file should raise."""
        with pytest.raises(FileNotFoundError):
            PrecursorClassifier.load(tmp_path / "no_such_file.json")


# ------------------------------------------------------------------ #
#  Training data generation tests                                     #
# ------------------------------------------------------------------ #

class TestLandslideTrainingData:
    """Tests for synthetic training data generation."""

    def test_correct_sample_counts(self):
        """Generated dataset should have the right number of samples."""
        gen = LandslideTrainingData(seed=42)
        X, y = gen.generate_synthetic_training_set(n_positive=50, n_negative=150)
        assert X.shape[0] == 200
        assert y.shape[0] == 200
        assert int(y.sum()) == 50
        assert int((1 - y).sum()) == 150

    def test_feature_matrix_shape(self):
        """Feature matrix width should match FEATURE_NAMES."""
        gen = LandslideTrainingData(seed=42)
        X, y = gen.generate_synthetic_training_set(n_positive=20, n_negative=80)
        assert X.shape[1] == len(FEATURE_NAMES)

    def test_all_features_finite(self):
        """No NaN/Inf in generated features."""
        gen = LandslideTrainingData(seed=42)
        X, y = gen.generate_synthetic_training_set(n_positive=50, n_negative=200)
        assert np.all(np.isfinite(X))

    def test_deterministic_with_seed(self):
        """Same seed should produce identical data."""
        gen1 = LandslideTrainingData(seed=123)
        X1, y1 = gen1.generate_synthetic_training_set(n_positive=20, n_negative=80)

        gen2 = LandslideTrainingData(seed=123)
        X2, y2 = gen2.generate_synthetic_training_set(n_positive=20, n_negative=80)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


# ------------------------------------------------------------------ #
#  End-to-end train → predict pipeline                                #
# ------------------------------------------------------------------ #

class TestTrainPipeline:
    """Tests for the full train_precursor_model pipeline."""

    def test_train_and_load(self, tmp_path):
        """Pipeline should train, save, and produce loadable model."""
        path = tmp_path / "model.json"
        metrics = train_precursor_model(
            path,
            seed=42,
            n_positive=50,
            n_negative=200,
            n_iter=200,
        )

        assert path.is_file()
        assert metrics["accuracy"] > 0.5  # better than random
        assert metrics["precision"] > 0.0
        assert metrics["recall"] > 0.0

        # Should be loadable and produce valid predictions
        model = PrecursorClassifier.load(path)
        gen = LandslideTrainingData(seed=99)
        X, y = gen.generate_synthetic_training_set(n_positive=10, n_negative=40)
        proba = model.predict_proba(X)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_train_achieves_high_accuracy(self, tmp_path):
        """Pipeline with CLI defaults should achieve >80% accuracy and >50% recall."""
        path = tmp_path / "model.json"
        metrics = train_precursor_model(
            path,
            seed=42,
            n_positive=200,
            n_negative=800,
            n_iter=500,
            lr=0.1,
        )

        assert metrics["accuracy"] > 0.80
        assert metrics["recall"] > 0.50

    def test_trained_model_discriminates(self, tmp_path):
        """Model should give higher scores to positive examples on average."""
        path = tmp_path / "model.json"
        train_precursor_model(
            path,
            seed=42,
            n_positive=100,
            n_negative=400,
            n_iter=300,
        )
        model = PrecursorClassifier.load(path)

        # Generate fresh test data with different seed
        gen = LandslideTrainingData(seed=999)
        X, y = gen.generate_synthetic_training_set(n_positive=30, n_negative=120)
        proba = model.predict_proba(X)

        mean_pos = np.mean(proba[y == 1])
        mean_neg = np.mean(proba[y == 0])
        assert mean_pos > mean_neg


# ------------------------------------------------------------------ #
#  Sigmoid edge cases                                                 #
# ------------------------------------------------------------------ #

class TestSigmoid:
    """Tests for the numerically stable sigmoid."""

    def test_sigmoid_large_positive(self):
        """Very large positive input should be ~1, not overflow."""
        result = _sigmoid(np.array([1000.0]))
        assert np.isfinite(result[0])
        assert result[0] == pytest.approx(1.0, abs=1e-10)

    def test_sigmoid_large_negative(self):
        """Very large negative input should be ~0, not overflow."""
        result = _sigmoid(np.array([-1000.0]))
        assert np.isfinite(result[0])
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_sigmoid_zero(self):
        """sigmoid(0) == 0.5."""
        result = _sigmoid(np.array([0.0]))
        assert result[0] == pytest.approx(0.5)


# ------------------------------------------------------------------ #
#  CLI train command                                                   #
# ------------------------------------------------------------------ #

class TestTrainCLI:
    """Tests for the glews train CLI command."""

    def test_train_command_runs(self, tmp_path):
        """The train command should produce a model file."""
        from click.testing import CliRunner
        from glews.cli import main

        output = tmp_path / "model.json"
        runner = CliRunner()
        result = runner.invoke(main, [
            "train",
            "--output", str(output),
            "--seed", "42",
            "--n-positive", "50",
            "--n-negative", "200",
            "--n-iter", "100",
        ])

        assert result.exit_code == 0, result.output
        assert output.is_file()
        assert "Training complete" in result.output

    def test_train_command_creates_directory(self, tmp_path):
        """The train command should create the output directory if needed."""
        from click.testing import CliRunner
        from glews.cli import main

        output = tmp_path / "subdir" / "model.json"
        runner = CliRunner()
        result = runner.invoke(main, [
            "train",
            "--output", str(output),
            "--seed", "42",
            "--n-positive", "30",
            "--n-negative", "100",
            "--n-iter", "50",
        ])

        assert result.exit_code == 0, result.output
        assert output.is_file()
