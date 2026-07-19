"""
Quick smoke test — runs each module independently with synthetic data
so you can verify the code works even without network.
"""
import os
import sys
import numpy as np
import pandas as pd

# Make module imports work whether run from /src or /
sys.path.insert(0, os.path.dirname(__file__))


def make_fake_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic 4h OHLCV series with a clear uptrend."""
    rng = np.random.default_rng(seed)
    n_ = n
    close = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.02, n_)))
    high = close * (1 + rng.uniform(0.001, 0.01, n_))
    low = close * (1 - rng.uniform(0.001, 0.01, n_))
    open_ = close + rng.normal(0, 0.5, n_)
    volume = rng.uniform(1000, 5000, n_)
    quote_volume = volume * close
    times = pd.date_range("2024-01-01", periods=n_, freq="4h", tz="UTC")
    return pd.DataFrame({
        "open_time": times,
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume, "quote_volume": quote_volume,
    })


def test_technical():
    from technical import technical_analysis
    df = make_fake_ohlcv()
    res = technical_analysis(df)
    print("Technical analysis OK:")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    assert 0 <= res["technical_score"] <= 100


def test_ml():
    from ml_weights import (
        compute_final_score, social_score_from_metrics, whale_score_from_features,
        append_history, WeightTuner, _load_history, HISTORY_COLS,
    )
    base_w = {"technical": 40, "pattern": 20, "social": 25, "whale": 15}
    s = compute_final_score(base_w, 80, 70, 60, 50)
    print(f"Final score with base weights: {s:.1f}")
    assert 0 <= s <= 100

    # Append some fake history
    os.makedirs("data", exist_ok=True)
    test_path = "data/smoke_history.csv"
    if os.path.exists(test_path):
        os.remove(test_path)
    # Patch history_path used inside ml_weights
    import ml_weights
    ml_weights.HISTORY_COLS = HISTORY_COLS
    for i in range(30):
        append_history(test_path, {
            "timestamp": f"2024-01-01T00:00:{i:02d}",
            "symbol": f"TEST{i}USDT",
            "market_cap_usd": 5e6,
            "technical": 70 + i % 10,
            "pattern": 60,
            "social": 75,
            "whale": 50,
            "rsi_value": 55,
            "macd_hist": 0.01,
            "social_score": 80,
            "liq_bias": 0.2,
            "price_change_pct_24h": 2.0,
            "score": 80 if i % 3 else 50,
            "w_technical": 40, "w_pattern": 20, "w_social": 25, "w_whale": 15,
        })
    df = _load_history(test_path)
    print(f"History rows: {len(df)}")
    # Manually label
    df["label"] = (df["score"] >= 80).astype(int)
    df.to_csv(test_path, index=False)

    tuner = WeightTuner("models/smoke_model.joblib", test_path, min_train=20)
    assert tuner.has_enough_data()
    ok = tuner.train()
    print(f"ML train: {ok}")
    new_w = tuner.suggest_weights(base_w)
    print(f"Suggested weights: {new_w}")
    total = sum(new_w.values())
    print(f"Sum: {total:.2f}")
    assert abs(total - 100) < 0.5


def test_social_whale():
    from ml_weights import social_score_from_metrics, whale_score_from_features
    s = social_score_from_metrics(
        {"galaxy_score": 80, "sentiment": 0.6, "social_dominance": 5},
        {"rising": True, "avg": 30},
        {"buy_ratio": 0.65, "idea_count": 10},
    )
    w = whale_score_from_features({"liq_bias": 0.4})
    print(f"Social: {s:.1f}, Whale: {w:.1f}")
    assert 0 <= s <= 100 and 0 <= w <= 100


def test_dashboard():
    from dashboard import render_dashboard
    # Create a fake last_scan.json
    fake = {
        "timestamp": "2024-01-01T00:00:00+00:00",
        "weights": {"technical": 40, "pattern": 20, "social": 25, "whale": 15},
        "threshold": 85,
        "scanned": 3,
        "alerts_count": 1,
        "alerts": [{
            "symbol": "TESTUSDT", "score": 92.0, "market_cap_usd": 1e7,
            "quote_volume_24h": 5e6, "rsi_value": 28, "rsi_divergence": "bullish_div",
            "macd_hist": 0.012, "macd_divergence": "none", "ema_alignment": "bullish",
            "patterns": ["bullish_engulfing", "double_bottom"],
            "social_score": 75, "liq_bias": 0.3, "price_change_pct_24h": 4.5,
        }],
        "results": [
            {"symbol": "TESTUSDT", "score": 92.0, "market_cap_usd": 1e7,
             "quote_volume_24h": 5e6, "rsi_value": 28, "rsi_divergence": "bullish_div",
             "macd_hist": 0.012, "macd_divergence": "none", "ema_alignment": "bullish",
             "patterns": ["bullish_engulfing", "double_bottom"],
             "social_score": 75, "liq_bias": 0.3, "price_change_pct_24h": 4.5},
            {"symbol": "FOOUSDT", "score": 76.0, "market_cap_usd": 5e6,
             "quote_volume_24h": 2e6, "rsi_value": 55, "rsi_divergence": "none",
             "macd_hist": 0.001, "macd_divergence": "none", "ema_alignment": "mixed",
             "patterns": [], "social_score": 60, "liq_bias": 0.0,
             "price_change_pct_24h": 1.0},
            {"symbol": "BARUSDT", "score": 55.0, "market_cap_usd": 3e6,
             "quote_volume_24h": 1.5e6, "rsi_value": 70, "rsi_divergence": "bearish_div",
             "macd_hist": -0.005, "macd_divergence": "bearish_div",
             "ema_alignment": "bearish", "patterns": ["bearish_engulfing"],
             "social_score": 40, "liq_bias": -0.4, "price_change_pct_24h": -2.0},
        ],
    }
    os.makedirs("data", exist_ok=True)
    import json
    with open("data/last_scan.json", "w", encoding="utf-8") as f:
        json.dump(fake, f, ensure_ascii=False, indent=2)
    p = render_dashboard()
    print(f"Dashboard rendered to {p} (size={os.path.getsize(p)} bytes)")


if __name__ == "__main__":
    print("=== Technical analysis ===")
    test_technical()
    print("\n=== ML weights ===")
    test_ml()
    print("\n=== Social/Whale ===")
    test_social_whale()
    print("\n=== Dashboard ===")
    test_dashboard()
    print("\n✅ All smoke tests passed.")
