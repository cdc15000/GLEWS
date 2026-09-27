"""
Transfer learning classifier for pre-failure pattern detection.

Implements a lightweight ML pipeline (numpy only) that learns to
distinguish displacement time series exhibiting pre-failure
acceleration (Voight-like exponential runaway) from benign patterns
(seasonal motion, steady creep, transient speedups).

The classifier is trained on synthetic data generated from known
failure kinematics, then applied as an optional scoring layer in
the anomaly detection pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Canonical feature order — load-bearing for standardization and
# save/load round-trips.  Never reorder without bumping MODEL_VERSION.
FEATURE_NAMES: list[str] = [
    "velocity_mean",
    "velocity_max",
    "velocity_std",
    "acceleration_mean",
    "acceleration_max",
    "acceleration_std",
    "velocity_trend",
    "acceleration_trend",
    "velocity_ratio",          # max velocity / baseline velocity
    "threshold_crossings",     # count of velocity crossings above k·std
    "temporal_asymmetry",      # fraction of total accel in last 30%
    "total_displacement",
    "anomalous_accel_duration",  # fraction of series with |accel| > k·std
]

MODEL_VERSION = 1


# ------------------------------------------------------------------ #
#  Feature extraction                                                 #
# ------------------------------------------------------------------ #

class PrecursorFeatureExtractor:
    """
    Extracts a fixed-length feature vector from a displacement time
    series for pre-failure pattern classification.

    Parameters
    ----------
    baseline_fraction : float
        Fraction of the early series used to define the baseline
        velocity (for the velocity_ratio feature).
    threshold_k : float
        Number of baseline standard deviations used to define
        "anomalous" velocity / acceleration for counting crossings
        and duration.
    """

    def __init__(
        self,
        baseline_fraction: float = 0.3,
        threshold_k: float = 2.0,
    ) -> None:
        self.baseline_fraction = baseline_fraction
        self.threshold_k = threshold_k

    def extract_features(
        self,
        dates: np.ndarray,
        displacement: np.ndarray,
    ) -> np.ndarray:
        """
        Extract feature vector from a displacement time series.

        Parameters
        ----------
        dates : np.ndarray
            Ordinal days (monotonically increasing), shape [n].
        displacement : np.ndarray
            Displacement in meters, shape [n].

        Returns
        -------
        np.ndarray
            Feature vector of length ``len(FEATURE_NAMES)``.
        """
        dates = np.asarray(dates, dtype=float)
        displacement = np.asarray(displacement, dtype=float)

        # Remove NaN entries
        valid = np.isfinite(dates) & np.isfinite(displacement)
        dates = dates[valid]
        displacement = displacement[valid]

        n = len(dates)
        if n < 3:
            return np.zeros(len(FEATURE_NAMES))

        # Time in years for velocity/acceleration units
        dt_days = np.diff(dates)
        dt_days[dt_days == 0] = 1.0
        dt_yr = dt_days / 365.25

        # Velocity (m/yr) and acceleration (m/yr^2)
        velocity = np.diff(displacement) / dt_yr
        if len(velocity) < 2:
            acceleration = np.zeros(1)
        else:
            dt_yr_a = (dt_yr[:-1] + dt_yr[1:]) / 2.0
            dt_yr_a[dt_yr_a == 0] = 1.0
            acceleration = np.diff(velocity) / dt_yr_a

        # --- Basic statistics ---
        vel_mean = np.mean(np.abs(velocity))
        vel_max = np.max(np.abs(velocity))
        vel_std = np.std(velocity)

        acc_mean = np.mean(np.abs(acceleration))
        acc_max = np.max(np.abs(acceleration))
        acc_std = np.std(acceleration)

        # --- Trend (linear slope of velocity / acceleration over time) ---
        vel_times = (dates[:-1] + dates[1:]) / 2.0
        velocity_trend = self._linear_slope(vel_times, np.abs(velocity))

        if len(acceleration) >= 2:
            acc_times = vel_times[:-1]
            if len(acc_times) == len(acceleration):
                acceleration_trend = self._linear_slope(
                    acc_times, np.abs(acceleration)
                )
            else:
                acceleration_trend = 0.0
        else:
            acceleration_trend = 0.0

        # --- Velocity ratio: max velocity / baseline velocity ---
        n_baseline = max(2, int(self.baseline_fraction * len(velocity)))
        baseline_vel = np.abs(velocity[:n_baseline])
        baseline_mean = np.mean(baseline_vel) if len(baseline_vel) > 0 else 0.0
        if baseline_mean > 1e-10:
            velocity_ratio = vel_max / baseline_mean
        else:
            velocity_ratio = 0.0

        # --- Threshold crossings ---
        baseline_std = np.std(baseline_vel) if len(baseline_vel) > 1 else 0.0
        if baseline_std > 1e-10 and baseline_mean > 1e-10:
            thresh = baseline_mean + self.threshold_k * baseline_std
            above = np.abs(velocity) > thresh
            # Count transitions from below to above
            crossings = int(np.sum(np.diff(above.astype(int)) > 0))
        else:
            crossings = 0

        # --- Temporal asymmetry ---
        # Fraction of total absolute acceleration concentrated in the
        # last 30% of the series.  Pre-failure patterns load late.
        total_abs_acc = np.sum(np.abs(acceleration))
        if total_abs_acc > 1e-10 and len(acceleration) >= 3:
            n_late = max(1, int(0.3 * len(acceleration)))
            late_acc = np.sum(np.abs(acceleration[-n_late:]))
            temporal_asymmetry = late_acc / total_abs_acc
        else:
            temporal_asymmetry = 0.0

        # --- Total displacement ---
        total_displacement = np.abs(displacement[-1] - displacement[0])

        # --- Duration of anomalous acceleration ---
        if baseline_std > 1e-10:
            acc_thresh = self.threshold_k * baseline_std
            anomalous_frac = np.mean(np.abs(acceleration) > acc_thresh)
        else:
            anomalous_frac = 0.0

        features = np.array([
            vel_mean,
            vel_max,
            vel_std,
            acc_mean,
            acc_max,
            acc_std,
            velocity_trend,
            acceleration_trend,
            velocity_ratio,
            float(crossings),
            temporal_asymmetry,
            total_displacement,
            anomalous_frac,
        ])

        # Guarantee finiteness — NaN/inf poison gradient descent
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        assert len(features) == len(FEATURE_NAMES)
        return features

    @staticmethod
    def _linear_slope(t: np.ndarray, y: np.ndarray) -> float:
        """Slope of a least-squares linear fit (robust to short series)."""
        if len(t) < 2:
            return 0.0
        t_c = t - t.mean()
        denom = np.sum(t_c ** 2)
        if denom < 1e-20:
            return 0.0
        return float(np.sum(t_c * y) / denom)


# ------------------------------------------------------------------ #
#  Logistic regression (numpy only)                                   #
# ------------------------------------------------------------------ #

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    # Clip to avoid overflow in exp
    z = np.clip(z, -500, 500)
    pos = z >= 0
    result = np.empty_like(z)
    result[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    result[~pos] = exp_z / (1.0 + exp_z)
    return result


class PrecursorClassifier:
    """
    Binary logistic regression classifier (numpy only).

    Learns to separate pre-failure displacement patterns from benign
    ones.  Includes feature standardization (z-score) as part of the
    model so that ``predict_proba`` works on raw features.
    """

    def __init__(self) -> None:
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.1,
        n_iter: int = 500,
        reg_lambda: float = 0.01,
    ) -> list[float]:
        """
        Train via gradient descent with L2 regularization.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape [n_samples, n_features].
        y : np.ndarray
            Binary labels (0/1), shape [n_samples].
        lr : float
            Learning rate.
        n_iter : int
            Number of gradient descent iterations.
        reg_lambda : float
            L2 regularization strength.

        Returns
        -------
        list[float]
            Loss at each iteration (for convergence diagnostics).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        # Standardize features
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-10] = 1.0  # avoid division by zero
        X_s = (X - self.mean) / self.std

        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        losses: list[float] = []
        for _ in range(n_iter):
            z = X_s @ self.weights + self.bias
            p = _sigmoid(z)

            # Binary cross-entropy + L2
            eps = 1e-15
            loss = -np.mean(
                y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
            ) + 0.5 * reg_lambda * np.sum(self.weights ** 2)
            losses.append(float(loss))

            # Gradients
            error = p - y
            dw = (X_s.T @ error) / n_samples + reg_lambda * self.weights
            db = np.mean(error)

            self.weights -= lr * dw
            self.bias -= lr * db

        return losses

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of pre-failure pattern.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape [n_samples, n_features] or [n_features].

        Returns
        -------
        np.ndarray
            Probabilities, shape [n_samples].
        """
        if self.weights is None or self.mean is None or self.std is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_s = (X - self.mean) / self.std
        z = X_s @ self.weights + self.bias
        return _sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Binary prediction.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        threshold : float
            Decision threshold.

        Returns
        -------
        np.ndarray
            Binary predictions (0 or 1), shape [n_samples].
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        """Serialize model to JSON."""
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet.")
        data = {
            "version": MODEL_VERSION,
            "feature_names": FEATURE_NAMES,
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> PrecursorClassifier:
        """Deserialize model from JSON."""
        data = json.loads(Path(path).read_text())
        model = cls()
        model.weights = np.array(data["weights"])
        model.bias = data["bias"]
        model.mean = np.array(data["mean"])
        model.std = np.array(data["std"])
        return model


# ------------------------------------------------------------------ #
#  Synthetic training data                                            #
# ------------------------------------------------------------------ #

class LandslideTrainingData:
    """
    Generates synthetic displacement time series for training.

    Positive class: exponential acceleration / Voight-like runaway.
    Negative class: seasonal motion, steady creep, transient speedups.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_synthetic_training_set(
        self,
        n_positive: int = 200,
        n_negative: int = 800,
        n_dates: int = 60,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate labeled feature matrix.

        Parameters
        ----------
        n_positive : int
            Number of positive (pre-failure) samples.
        n_negative : int
            Number of negative (benign) samples.
        n_dates : int
            Length of each synthetic time series.

        Returns
        -------
        X : np.ndarray
            Feature matrix, shape [n_positive + n_negative, n_features].
        y : np.ndarray
            Labels (1 = pre-failure, 0 = benign).
        """
        extractor = PrecursorFeatureExtractor()
        features_list: list[np.ndarray] = []
        labels: list[int] = []

        # Positive samples: exponential acceleration
        for _ in range(n_positive):
            dates, disp = self._generate_positive(n_dates)
            features_list.append(extractor.extract_features(dates, disp))
            labels.append(1)

        # Negative samples: mix of benign patterns
        n_seasonal = n_negative // 4
        n_creep = n_negative // 4
        n_transient = n_negative // 4
        n_quiet = n_negative - n_seasonal - n_creep - n_transient

        for _ in range(n_seasonal):
            dates, disp = self._generate_seasonal(n_dates)
            features_list.append(extractor.extract_features(dates, disp))
            labels.append(0)

        for _ in range(n_creep):
            dates, disp = self._generate_steady_creep(n_dates)
            features_list.append(extractor.extract_features(dates, disp))
            labels.append(0)

        for _ in range(n_transient):
            dates, disp = self._generate_transient_speedup(n_dates)
            features_list.append(extractor.extract_features(dates, disp))
            labels.append(0)

        for _ in range(n_quiet):
            dates, disp = self._generate_quiet(n_dates)
            features_list.append(extractor.extract_features(dates, disp))
            labels.append(0)

        X = np.vstack(features_list)
        y = np.array(labels)

        # Shuffle
        idx = self.rng.permutation(len(y))
        return X[idx], y[idx]

    def _generate_positive(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Exponential acceleration / Voight-like pre-failure."""
        dates = np.arange(n) * 12.0 + 738000  # 12-day revisit
        t_norm = np.linspace(0, 1, n)

        # Base creep
        base_rate = self.rng.uniform(0.01, 0.05)  # m/yr
        disp = base_rate * t_norm * (n * 12 / 365.25)

        # Exponential acceleration in latter half
        onset = self.rng.uniform(0.3, 0.7)
        amplitude = self.rng.uniform(0.1, 2.0)
        exp_rate = self.rng.uniform(3.0, 8.0)
        mask = t_norm > onset
        t_accel = (t_norm[mask] - onset) / (1 - onset)
        disp[mask] += amplitude * (np.exp(exp_rate * t_accel) - 1) / np.exp(exp_rate)

        # Noise
        noise_level = self.rng.uniform(0.002, 0.02)
        disp += self.rng.normal(0, noise_level, n)

        return dates, disp

    def _generate_seasonal(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Seasonal oscillation — benign."""
        dates = np.arange(n) * 12.0 + 738000
        t_days = dates - dates[0]

        amplitude = self.rng.uniform(0.01, 0.1)
        phase = self.rng.uniform(0, 2 * np.pi)
        disp = amplitude * np.sin(2 * np.pi * t_days / 365.25 + phase)

        # Small linear trend
        rate = self.rng.uniform(-0.01, 0.01)
        disp += rate * t_days / 365.25

        noise_level = self.rng.uniform(0.002, 0.01)
        disp += self.rng.normal(0, noise_level, n)
        return dates, disp

    def _generate_steady_creep(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Constant velocity — benign."""
        dates = np.arange(n) * 12.0 + 738000
        t_days = dates - dates[0]

        rate = self.rng.uniform(0.02, 0.15)  # m/yr — can be fast
        disp = rate * t_days / 365.25

        noise_level = self.rng.uniform(0.002, 0.015)
        disp += self.rng.normal(0, noise_level, n)
        return dates, disp

    def _generate_transient_speedup(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Transient velocity pulse that decays — the hard negative case.

        Looks like onset of failure but velocity returns to baseline.
        """
        dates = np.arange(n) * 12.0 + 738000
        t_norm = np.linspace(0, 1, n)

        base_rate = self.rng.uniform(0.01, 0.05)
        disp = base_rate * t_norm * (n * 12 / 365.25)

        # Transient pulse: rises then decays
        center = self.rng.uniform(0.3, 0.7)
        width = self.rng.uniform(0.05, 0.15)
        amplitude = self.rng.uniform(0.05, 0.3)
        pulse = amplitude * np.exp(-0.5 * ((t_norm - center) / width) ** 2)
        disp += np.cumsum(pulse) * 12 / 365.25

        noise_level = self.rng.uniform(0.002, 0.015)
        disp += self.rng.normal(0, noise_level, n)
        return dates, disp

    def _generate_quiet(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Near-zero displacement — benign."""
        dates = np.arange(n) * 12.0 + 738000

        noise_level = self.rng.uniform(0.001, 0.005)
        disp = self.rng.normal(0, noise_level, n)
        return dates, disp


# ------------------------------------------------------------------ #
#  Training entry point                                               #
# ------------------------------------------------------------------ #

def train_precursor_model(
    output_path: str | Path,
    seed: int = 42,
    n_positive: int = 200,
    n_negative: int = 800,
    n_iter: int = 500,
    lr: float = 0.1,
) -> dict:
    """
    Generate training data, train, evaluate, and save the model.

    Parameters
    ----------
    output_path : str or Path
        Where to save the trained model JSON.
    seed : int
        Random seed for reproducibility.
    n_positive, n_negative : int
        Class sizes.
    n_iter : int
        Gradient descent iterations.
    lr : float
        Learning rate.

    Returns
    -------
    dict
        Evaluation metrics: accuracy, precision, recall on held-out split.
    """
    generator = LandslideTrainingData(seed=seed)
    X, y = generator.generate_synthetic_training_set(
        n_positive=n_positive, n_negative=n_negative,
    )

    # 80/20 train/test split (deterministic via seed)
    rng = np.random.default_rng(seed + 1)
    n_total = len(y)
    idx = rng.permutation(n_total)
    n_train = int(0.8 * n_total)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = PrecursorClassifier()
    losses = model.fit(X_train, y_train, lr=lr, n_iter=n_iter)

    # Evaluate
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    accuracy = float(np.mean(preds == y_test))

    tp = float(np.sum((preds == 1) & (y_test == 1)))
    fp = float(np.sum((preds == 1) & (y_test == 0)))
    fn = float(np.sum((preds == 0) & (y_test == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    model.save(output_path)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "final_loss": losses[-1] if losses else float("nan"),
        "n_train": n_train,
        "n_test": n_total - n_train,
    }

    logger.info(
        "Precursor model trained: acc=%.3f, prec=%.3f, rec=%.3f, "
        "loss=%.4f → %s",
        accuracy, precision, recall, metrics["final_loss"], output_path,
    )

    return metrics
