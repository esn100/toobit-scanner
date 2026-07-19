"""
ML engine (Layer 6 + 7 + 9 of the PumpHunter pipeline).

Components:
  - Chi-square feature significance test on labeled history
  - Logistic regression model predicting P(success)
  - Soft weight adaptation: when new data arrives, blend the current
    weights with feature importances derived from logistic regression
    coefficients
  - Persistence to disk
  - Train/test split + simple metrics
"""
from __future__ import annotations
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import chi2_contingency

from features import FEATURE_NAMES, feature_vector


HISTORY_COLS = (
    ["timestamp", "symbol", "label", "score", "ml_prob",
     "outcome_score", "composite_score"]
    + FEATURE_NAMES
)


# ----------------------------------------------------------------------------
# 1) History helpers
# ----------------------------------------------------------------------------
def _load_history(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame(columns=HISTORY_COLS)
    return pd.DataFrame(columns=HISTORY_COLS)


def append_signal_history(path: str, row: Dict) -> None:
    df = _load_history(path)
    for c in HISTORY_COLS:
        if c not in df.columns:
            df[c] = np.nan
    row_aligned = {c: row.get(c, np.nan) for c in HISTORY_COLS}
    new_row = pd.DataFrame([row_aligned])
    # Drop all-NA columns from df to avoid the FutureWarning
    df_clean = df[HISTORY_COLS].dropna(axis=1, how="all")
    keep_cols = [c for c in HISTORY_COLS if c in df_clean.columns]
    new_aligned = new_row.reindex(columns=keep_cols)
    df_clean = pd.concat([df_clean, new_aligned], ignore_index=True)
    # Re-add any all-NA columns that may have been dropped
    for c in HISTORY_COLS:
        if c not in df_clean.columns:
            df_clean[c] = np.nan
    df_clean = df_clean.tail(10000)
    df_clean.to_csv(path, index=False)


def update_signal_outcome(
    path: str,
    symbol: str,
    timestamp: str,
    outcome_score: float,
    label: int,
) -> bool:
    """Mark a previously stored signal with its outcome."""
    df = _load_history(path)
    if df.empty:
        return False
    mask = (df["symbol"] == symbol) & (df["timestamp"] == timestamp) & df["label"].isna()
    if not mask.any():
        return False
    df.loc[mask, "outcome_score"] = float(outcome_score)
    df.loc[mask, "label"] = int(label)
    df.to_csv(path, index=False)
    return True


# ----------------------------------------------------------------------------
# 2) Chi-square significance test
# ----------------------------------------------------------------------------
def chi_square_significance(
    history: pd.DataFrame,
    features: List[str],
    min_samples: int = 30,
) -> Dict[str, Dict]:
    """
    For each feature, test whether its high/low presence is associated
    with successful signals. We bin each feature by median and run
    a 2x2 chi-square test.

    Returns a dict of {feature: {chi2, p_value, significant, n}}.
    """
    out: Dict[str, Dict] = {}
    df = history.dropna(subset=["label"]).copy()
    if len(df) < min_samples or df["label"].nunique() < 2:
        return out
    for f in features:
        if f not in df.columns:
            continue
        s = df[f]
        if s.nunique() < 2:
            continue
        try:
            threshold = float(s.median())
            hi = s >= threshold
            lo = ~hi
            # contingency: high/low x success/fail
            table = np.array([
                [(df["label"][hi] == 1).sum(), (df["label"][hi] == 0).sum()],
                [(df["label"][lo] == 1).sum(), (df["label"][lo] == 0).sum()],
            ])
            if table.min() == 0 or table.sum() == 0:
                continue
            chi2, p, dof, _ = chi2_contingency(table)
            out[f] = {
                "chi2": float(chi2),
                "p_value": float(p),
                "significant": bool(p < 0.05),
                "n": int(len(df)),
            }
        except Exception:
            continue
    return out


# ----------------------------------------------------------------------------
# 3) Logistic regression
# ----------------------------------------------------------------------------
class PumpHunterML:
    """
    Logistic regression that predicts P(success | features).
    """
    def __init__(self, model_path: str, min_train: int = 30):
        self.model_path = model_path
        self.min_train = min_train
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.model_path):
            try:
                bundle = joblib.load(self.model_path)
                self.model = bundle["model"]
                self.scaler = bundle["scaler"]
            except Exception:
                self.model = None
                self.scaler = None

    def has_enough_data(self, history_path: str | None = None) -> bool:
        if history_path is None:
            # Default: data/signal_history.csv relative to model_path
            base = os.path.dirname(self.model_path)
            # models/foo.joblib -> ../data/signal_history.csv
            history_path = os.path.abspath(
                os.path.join(base, "..", "data", "signal_history.csv")
            )
        df = _load_history(history_path)
        df = df.dropna(subset=["label"])
        return len(df) >= self.min_train and df["label"].nunique() == 2

    def train(self, history_path: str) -> bool:
        df = _load_history(history_path)
        df = df.dropna(subset=["label"])
        if len(df) < self.min_train or df["label"].nunique() < 2:
            return False
        X, y = [], []
        for _, row in df.iterrows():
            feats = {f: row.get(f, 0.0) for f in FEATURE_NAMES}
            X.append(feature_vector(feats))
            y.append(int(row["label"]))
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        # Stratified split
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        self.scaler = StandardScaler()
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=0.5,
            random_state=42,
        )
        self.model.fit(X_tr_s, y_tr)
        # Save
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        joblib.dump(
            {"model": self.model, "scaler": self.scaler,
             "features": FEATURE_NAMES},
            self.model_path,
        )
        # Persist metrics
        preds = self.model.predict(X_te_s)
        try:
            proba = self.model.predict_proba(X_te_s)[:, 1]
        except Exception:
            proba = preds.astype(float)
        metrics = {
            "timestamp": time.time(),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "accuracy": float(accuracy_score(y_te, preds)),
            "f1": float(f1_score(y_te, preds, zero_division=0)),
            "mean_proba": float(np.mean(proba)),
        }
        try:
            metrics_path = self.model_path.replace(".joblib", "_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        except Exception:
            pass
        return True

    def predict_proba(self, features: Dict[str, float]) -> float:
        if self.model is None or self.scaler is None:
            return 0.5
        x = feature_vector(features).reshape(1, -1)
        try:
            return float(self.model.predict_proba(self.scaler.transform(x))[0, 1])
        except Exception:
            return 0.5


# ----------------------------------------------------------------------------
# 4) Soft weight adaptation
# ----------------------------------------------------------------------------
# Map each config weight key to one or more feature names whose
# coefficients influence it.
WEIGHT_TO_FEATURES: Dict[str, List[str]] = {
    "technical": ["rsi_value", "rsi_divergence", "macd_hist",
                  "macd_divergence", "ema_alignment"],
    "momentum": ["momentum_1_pct", "momentum_3_pct", "momentum_6_pct",
                 "momentum_12_pct", "momentum_acceleration"],
    "volume": ["rvol", "volume_spike"],
    "vwap": ["vwap_distance_pct", "price_above_vwap"],
    "atr_bb": ["atr_pct", "atr_expanding", "bb_squeeze", "bb_breakout_above"],
    "structure": ["higher_highs", "higher_lows", "bos_up", "in_range"],
    "candle": ["candle_strength", "big_wick_top", "power_streak"],
    "mtf": ["mtf_alignment"],
    "pattern": ["rsi_divergence", "macd_divergence", "candle_strength",
                "power_streak"],
}


def soft_adapt_weights(
    base_weights: Dict[str, float],
    ml: PumpHunterML,
    blend: float = 0.3,
) -> Dict[str, float]:
    """
    Adjust the rule-based weights by blending with logistic regression
    coefficient magnitudes. A higher blend (e.g. 0.5) trusts the ML
    model more. Returns a new dict whose values sum to 100.
    """
    if ml.model is None or ml.scaler is None:
        return base_weights
    try:
        coefs = np.abs(ml.model.coef_[0])
    except Exception:
        return base_weights
    if len(coefs) != len(FEATURE_NAMES):
        return base_weights
    feat_imp = dict(zip(FEATURE_NAMES, coefs))

    raw: Dict[str, float] = {}
    for wkey, fnames in WEIGHT_TO_FEATURES.items():
        s = sum(feat_imp.get(f, 0.0) for f in fnames)
        raw[wkey] = s
    total = sum(raw.values()) or 1.0
    learned = {k: 100.0 * v / total for k, v in raw.items()}

    # Blend with prior, clamp to [3, 35], renormalise
    out = {}
    for k in base_weights:
        prior = float(base_weights.get(k, 10.0))
        l = float(learned.get(k, prior))
        merged = (1.0 - blend) * prior + blend * l
        out[k] = max(3.0, min(35.0, merged))
    s = sum(out.values()) or 1.0
    return {k: round(100.0 * v / s, 2) for k, v in out.items()}
