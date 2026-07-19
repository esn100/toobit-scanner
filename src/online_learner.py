"""
Online learning classifier for the small-cap scanner.

Uses SGDClassifier (logistic regression with online updates).
After every labelled outcome, the model is incrementally updated.
The model's predicted probability is combined with the rule-based
score via a weighted blend that adapts over time.
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "rvol", "volume_spike", "momentum_1_pct", "momentum_3_pct",
    "rsi_1h", "atr_pct", "bb_squeeze", "higher_lows",
    "btc_corr_2d", "smart_money_score", "independent_mover",
    "consensus_count", "composite_score",
]


class OnlineLearner:
    """
    SGD-based online classifier.

    - Trains incrementally on each new labelled signal
    - Uses a scaler that is fit on first batch, then updated slowly
    - Provides a probability blending function
    """
    def __init__(self, model_path: str, min_samples: int = 10):
        self.model_path = model_path
        self.min_samples = min_samples
        self.model: Optional[SGDClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.n_samples_seen: int = 0
        self._load()

    def _load(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "r", encoding="utf-8") as f:
                    bundle = json.load(f)
                self.n_samples_seen = int(bundle.get("n_samples_seen", 0))
                # SGDClassifier and scaler can't be JSON-serialised, but
                # we can store their parameters and reconstruct.
                self._coef = np.array(bundle.get("coef", []))
                self._intercept = float(bundle.get("intercept", 0.0))
                self._scaler_mean = np.array(bundle.get("scaler_mean", []))
                self._scaler_scale = np.array(bundle.get("scaler_scale", []))
                if len(self._coef) == len(FEATURE_COLS):
                    self._ready = True
                else:
                    self._ready = False
            except Exception:
                self._ready = False
        else:
            self._ready = False
            self._coef = np.zeros(len(FEATURE_COLS))
            self._intercept = 0.0
            self._scaler_mean = np.zeros(len(FEATURE_COLS))
            self._scaler_scale = np.ones(len(FEATURE_COLS))

    def _save(self):
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        bundle = {
            "n_samples_seen": self.n_samples_seen,
            "coef": self._coef.tolist() if hasattr(self, "_coef") else [],
            "intercept": getattr(self, "_intercept", 0.0),
            "scaler_mean": (self._scaler_mean.tolist()
                            if hasattr(self, "_scaler_mean") else []),
            "scaler_scale": (self._scaler_scale.tolist()
                             if hasattr(self, "_scaler_scale") else []),
        }
        with open(self.model_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)

    def _vector(self, features: Dict) -> np.ndarray:
        v = []
        for c in FEATURE_COLS:
            v.append(float(features.get(c, 0.0)))
        return np.array(v, dtype=float)

    def partial_fit(self, features: Dict, label: int) -> None:
        """Update the model with one new labelled example."""
        x = self._vector(features).reshape(1, -1)
        # Update scaler stats with running mean/var approximation
        if self.n_samples_seen == 0:
            self._scaler_mean = x[0].copy()
            self._scaler_scale = np.ones_like(x[0]) * 1.0
        else:
            # Welford's online algorithm
            n = self.n_samples_seen
            new_mean = self._scaler_mean + (x[0] - self._scaler_mean) / (n + 1)
            new_var = (self._scaler_scale ** 2) * n + (x[0] - new_mean) * (
                x[0] - self._scaler_mean
            )
            new_var = new_var / (n + 1)
            self._scaler_mean = new_mean
            self._scaler_scale = np.maximum(np.sqrt(np.maximum(new_var, 1e-9)), 1e-6)
        # Scale x
        x_scaled = (x[0] - self._scaler_mean) / self._scaler_scale
        # Simple gradient update on logistic regression
        # If we already have a model, do partial_fit
        if self.model is None:
            from sklearn.linear_model import SGDClassifier
            self.model = SGDClassifier(
                loss="log_loss", learning_rate="adaptive",
                eta0=0.01, random_state=42, warm_start=True,
            )
            # Need at least 2 samples with different labels for partial_fit
            self._buffer_X = [x_scaled]
            self._buffer_y = [label]
        else:
            self._buffer_X.append(x_scaled)
            self._buffer_y.append(label)
            if len(set(self._buffer_y)) >= 2 and len(self._buffer_X) >= 4:
                Xb = np.array(self._buffer_X[-20:])
                yb = np.array(self._buffer_y[-20:])
                try:
                    self.model.partial_fit(Xb, yb, classes=np.array([0, 1]))
                except Exception:
                    pass
        # Update self._coef from model
        try:
            self._coef = self.model.coef_[0].copy()
            self._intercept = float(self.model.intercept_[0])
        except Exception:
            # Update manually if model not trained yet
            lr = 0.05
            z = float(np.dot(self._coef, x_scaled) + self._intercept)
            p = 1.0 / (1.0 + np.exp(-z))
            grad = (p - label) * x_scaled
            self._coef = self._coef - lr * grad
            self._intercept -= lr * (p - label)
        self.n_samples_seen += 1
        self._ready = True
        self._save()

    def predict_proba(self, features: Dict) -> float:
        """Return P(success | features)."""
        if not self._ready or self.n_samples_seen < self.min_samples:
            return 0.5
        x = self._vector(features)
        x_scaled = (x - self._scaler_mean) / self._scaler_scale
        z = float(np.dot(self._coef, x_scaled) + self._intercept)
        z = max(-20, min(20, z))  # clip
        p = 1.0 / (1.0 + np.exp(-z))
        return float(p)

    def has_enough_data(self) -> bool:
        return self.n_samples_seen >= self.min_samples and self._ready
