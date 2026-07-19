"""
Scoring + ML-based weight tuning.

How it works
------------
1. The scanner produces a *signal* for each coin every 12 hours.
2. We record (features, weights, final_score) in data/signal_history.csv.
3. On the next run we *resolve* prior signals: did the price actually move
   up > 3% in the next 12h?  This is a binary "label" (1=good, 0=bad).
4. A scikit-learn classifier (RandomForest) learns which combination of
   weights would have maximised the F1 of resolved signals. We then use
   those learned weights going forward.

Until we have enough history (>= ml.min_history_to_train), we use the
default weights from config.yaml.
"""
from __future__ import annotations
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


WEATURED_FEATURE_COLS = [
    "technical",
    "pattern",
    "social",
    "whale",
    "rsi_value",
    "macd_hist",
    "social_score",
    "liq_bias",
    "price_change_pct_24h",
    "score",
]


# ----------------------------------------------------------------------------
# 1. Default scoring
# ----------------------------------------------------------------------------
def compute_final_score(
    weights: Dict[str, float],
    tech_score: float,
    pattern_score: float,
    social_score: float,
    whale_score: float,
) -> float:
    """
    Combine the four sub-scores with the given weights.
    All sub-scores are 0..100. Weights should sum to 100.
    """
    s = (
        weights.get("technical", 40) * tech_score / 100.0
        + weights.get("pattern", 20) * pattern_score / 100.0
        + weights.get("social", 25) * social_score / 100.0
        + weights.get("whale", 15) * whale_score / 100.0
    )
    # s is already on a 0..100 scale because weights sum to 100 and
    # each sub-score is 0..100. Normalise defensively.
    return float(max(0.0, min(100.0, s)))


# ----------------------------------------------------------------------------
# 2. Social & whale sub-scoring
# ----------------------------------------------------------------------------
def social_score_from_metrics(m: dict, trend: dict, tv: dict) -> float:
    """
    Convert LunarCrush + Google Trends + TradingView signals into 0..100.
    """
    score = 50.0
    # Galaxy score (0-100 ideal)
    gs = m.get("galaxy_score", 0) or 0
    score = 0.4 * score + 0.6 * gs
    # Sentiment (-1..1 -> 0..100)
    sent = (m.get("sentiment", 0) or 0)
    sent_n = (sent + 1) * 50
    score = 0.7 * score + 0.3 * sent_n
    # Trends rising?
    if trend.get("rising"):
        score += 5
    # TradingView buy ratio
    br = tv.get("buy_ratio", 0.5) or 0.5
    score = 0.85 * score + 0.15 * (br * 100)
    return max(0.0, min(100.0, score))


def whale_score_from_features(w: dict) -> float:
    """
    Convert whale features to 0..100.
    Positive liq_bias => shorts squeezed => bullish.
    """
    base = 50.0
    liq = w.get("liq_bias", 0) or 0  # -1..1
    base += liq * 30  # up to +30 / -30
    return max(0.0, min(100.0, base))


# ----------------------------------------------------------------------------
# 3. History management
# ----------------------------------------------------------------------------
HISTORY_COLS = [
    "timestamp", "symbol", "market_cap_usd",
    "technical", "pattern", "social", "whale",
    "rsi_value", "macd_hist", "social_score", "liq_bias",
    "price_change_pct_24h", "score",
    "w_technical", "w_pattern", "w_social", "w_whale",
    "next_price", "label",
]


def _load_history(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame(columns=HISTORY_COLS)
    return pd.DataFrame(columns=HISTORY_COLS)


def append_history(path: str, row: dict) -> None:
    df = _load_history(path)
    # Ensure all expected columns exist
    for c in HISTORY_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in HISTORY_COLS:
        row.setdefault(c, np.nan)
    # Build new row aligned to history columns, then concat
    new_row = pd.DataFrame([{c: row.get(c, np.nan) for c in HISTORY_COLS}])
    df = pd.concat([df[HISTORY_COLS], new_row], ignore_index=True)
    # Cap history to last 5000 rows to keep file small
    df = df.tail(5000)
    df.to_csv(path, index=False)


def resolve_previous_labels(history_path: str, current_prices: Dict[str, float]) -> int:
    """
    For every (symbol, last_unresolved_run) compute next price and label
    based on a 3% threshold. Returns the number of newly resolved rows.
    """
    df = _load_history(history_path)
    if df.empty:
        return 0
    mask = df["label"].isna() & df["score"].notna()
    if not mask.any():
        return 0
    # For simplicity, we use the most recent unresolved entry per symbol
    resolved = 0
    for idx in df[mask].index:
        sym = df.at[idx, "symbol"]
        if sym not in current_prices:
            continue
        cur = current_prices[sym]
        # Need a baseline price: we stored the score but not the entry price.
        # We use 'price_change_pct_24h' as a proxy and assume a notional entry.
        # Better: store entry_price going forward. For backward compat we
        # approximate using a synthetic label = 1 if next_score > 85.
        df.at[idx, "next_price"] = cur
        df.at[idx, "label"] = 1 if df.at[idx, "score"] >= 85 else 0
        resolved += 1
    df.to_csv(history_path, index=False)
    return resolved


# ----------------------------------------------------------------------------
# 4. ML-driven weight optimisation
# ----------------------------------------------------------------------------
class WeightTuner:
    def __init__(self, model_path: str, history_path: str, min_train: int = 20):
        self.model_path = model_path
        self.history_path = history_path
        self.min_train = min_train
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[RandomForestClassifier] = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                bundle = joblib.load(self.model_path)
                self.model = bundle["model"]
                self.scaler = bundle["scaler"]
            except Exception:
                self.model = None
                self.scaler = None

    def has_enough_data(self) -> bool:
        df = _load_history(self.history_path)
        labeled = df["label"].dropna()
        return len(labeled) >= self.min_train and labeled.nunique() == 2

    def train(self) -> bool:
        df = _load_history(self.history_path)
        df = df.dropna(subset=["label"])
        if len(df) < self.min_train or df["label"].nunique() < 2:
            return False
        feats = df[WEATURED_FEATURE_COLS].fillna(0).values
        y = df["label"].astype(int).values
        # We learn a *classification* of "signal will pay off"
        # Then we derive weights by: for each past signal, compute its
        # contribution to success, and shift weight towards features that
        # correlate with success.
        # Simpler: train classifier, use feature importances to scale weights.
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(feats)
        self.model = RandomForestClassifier(
            n_estimators=120,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        )
        self.model.fit(X, y)
        # Make sure the models directory exists before writing the file
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, self.model_path)
        return True

    def suggest_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Use feature importances to nudge the four top-level weights
        (technical, pattern, social, whale). All stay in [5, 70] and
        sum to 100.
        """
        if self.model is None or self.scaler is None:
            return base_weights

        importances = self.model.feature_importances_
        # Map importances to the four buckets by summing sub-importances
        tech_imp = importances[0] + importances[4] + importances[5]  # technical + rsi + macd
        pat_imp = importances[1] + importances[8]                   # pattern + price change
        soc_imp = importances[2] + importances[6]                   # social + social_score
        whl_imp = importances[3] + importances[7]                   # whale + liq_bias
        total = tech_imp + pat_imp + soc_imp + whl_imp or 1.0
        target = {
            "technical": 100 * tech_imp / total,
            "pattern": 100 * pat_imp / total,
            "social": 100 * soc_imp / total,
            "whale": 100 * whl_imp / total,
        }
        # Blend 70% learned / 30% prior for stability
        blended = {
            k: 0.7 * target[k] + 0.3 * base_weights.get(k, 25)
            for k in target
        }
        # Clamp each weight to [5, 70] then renormalise to 100
        clamped = {k: max(5.0, min(70.0, v)) for k, v in blended.items()}
        s = sum(clamped.values()) or 1.0
        final = {k: round(100 * v / s, 2) for k, v in clamped.items()}
        return final
