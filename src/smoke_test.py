"""
End-to-end smoke test for PumpHunter-AI.

Tests every layer without making any network calls.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def make_ohlcv(n: int = 300, seed: int = 42, trend: float = 0.001) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(trend, 0.02, n)))
    high = close * (1 + rng.uniform(0.001, 0.012, n))
    low = close * (1 - rng.uniform(0.001, 0.012, n))
    open_ = close + rng.normal(0, 0.5, n)
    vol = rng.uniform(1000, 5000, n)
    qv = vol * close
    t = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({
        "open_time": t, "open": open_, "high": high, "low": low,
        "close": close, "volume": vol, "quote_volume": qv,
    })


def test_data_quality():
    from data_quality import validate_ohlcv
    df = make_ohlcv()
    # Use a very long max_age so synthetic 2024 data isn't rejected
    rep = validate_ohlcv(df, max_age_hours=1_000_000)
    assert rep.ok, f"data quality fail: {rep.reasons}"
    print("✓ data_quality ok, candles:", rep.stats.get("candles"))

    # Bad df: negative price
    bad = df.copy()
    bad.loc[5, "close"] = -1
    rep = validate_ohlcv(bad)
    assert not rep.ok
    assert any("non-positive" in r for r in rep.reasons)
    print("✓ data_quality rejects negative prices")


def test_indicators():
    from indicators import (
        vwap_features, atr_features, bollinger_features,
        relative_volume, volume_continuity, momentum_features,
        mtf_alignment,
    )
    df = make_ohlcv()
    v = vwap_features(df)
    a = atr_features(df)
    b = bollinger_features(df)
    r = relative_volume(df)
    vc = volume_continuity(df)
    m = momentum_features(df)
    mt = mtf_alignment(df, df)
    for d in (v, a, b, r, vc, m, mt):
        assert isinstance(d, dict)
    print("✓ all indicator modules returned dicts")
    print(f"  vwap_score={v['vwap_score']:.1f}  rvol={r['rvol']:.2f}  "
          f"momentum_score={m['momentum_score']:.1f}  "
          f"m6={m['momentum_6_pct']:.2f}%")


def test_structure():
    from market_structure import structure_features
    df = make_ohlcv(trend=0.005)  # strong uptrend
    s = structure_features(df)
    assert s["structure_score"] >= 0
    print(f"✓ structure features: score={s['structure_score']:.1f} "
          f"higher_highs={s['higher_highs']} bos_up={s['bos_up']}")


def test_candle_quality():
    from candle_quality import candle_quality_features
    df = make_ohlcv()
    c = candle_quality_features(df)
    assert 0.0 <= c["candle_score"] <= 100.0
    print(f"✓ candle_quality: strength={c['candle_strength']:.2f} "
          f"score={c['candle_score']:.1f}")


def test_btc_filter():
    from btc_filter import BTCFilter
    from toobit_client import ToobitClient
    # Don't actually hit the network; just check the class
    b = BTCFilter(ToobitClient())
    print("✓ btc_filter instantiates")


def test_features():
    from features import build_features, feature_vector, FEATURE_NAMES
    feats = build_features(
        technical={"rsi_value": 35, "rsi_divergence": "bullish_div",
                   "macd_hist": 0.5, "macd_divergence": "none",
                   "ema_alignment": "bullish", "technical_score": 70,
                   "pattern_score": 60},
        indicators={"rvol": 2.5, "volume_spike": True, "vwap_score": 70,
                    "vwap_distance_pct": 1.2, "price_above_vwap": True,
                    "atr_score": 70, "atr_pct": 1.5, "atr_expanding": True,
                    "bb_score": 65, "bb_squeeze": True,
                    "bb_breakout_above": False, "momentum_score": 75,
                    "momentum_1_pct": 0.5, "momentum_3_pct": 1.5,
                    "momentum_6_pct": 3.0, "momentum_12_pct": 5.0,
                    "momentum_acceleration": 0.5},
        structure={"structure_score": 80, "higher_highs": True,
                   "higher_lows": True, "bos_up": True, "in_range": False},
        candle={"candle_score": 75, "candle_strength": 0.7,
                "big_wick_top": False, "power_streak": 2},
        mtf={"alignment_score": 75, "aligned": True, "same_sign": True,
             "fast_bias": 0.5, "slow_bias": 0.4},
        btc={"state": "BULLISH", "score_modifier": 1.05,
             "btc_momentum_12_pct": 2.0},
    )
    v = feature_vector(feats)
    assert len(v) == len(FEATURE_NAMES)
    print(f"✓ features: built {len(feats)} fields, vector len {len(v)}")


def test_scoring():
    from scoring import rule_based_score
    rb = rule_based_score(
        technical={"rsi_value": 35, "rsi_divergence": "bullish_div",
                   "macd_hist": 0.5, "macd_divergence": "none",
                   "ema_alignment": "bullish", "technical_score": 70,
                   "pattern_score": 60},
        indicators={"rvol": 2.5, "volume_spike": True, "vwap_score": 70,
                    "vwap_distance_pct": 1.2, "price_above_vwap": True,
                    "atr_score": 70, "atr_pct": 1.5, "atr_expanding": True,
                    "bb_score": 65, "bb_squeeze": True,
                    "bb_breakout_above": False, "momentum_score": 75,
                    "momentum_1_pct": 0.5, "momentum_3_pct": 1.5,
                    "momentum_6_pct": 3.0, "momentum_12_pct": 5.0,
                    "momentum_acceleration": 0.5},
        structure={"structure_score": 80, "higher_highs": True,
                   "higher_lows": True, "bos_up": True, "in_range": False},
        candle={"candle_score": 75, "candle_strength": 0.7,
                "big_wick_top": False, "power_streak": 2},
        mtf={"alignment_score": 75, "aligned": True, "same_sign": True,
             "fast_bias": 0.5, "slow_bias": 0.4},
        btc={"state": "BULLISH", "score_modifier": 1.05,
             "btc_momentum_12_pct": 2.0},
        weights={"technical": 12, "momentum": 12, "volume": 18, "vwap": 8,
                 "atr_bb": 6, "structure": 10, "candle": 8, "mtf": 8,
                 "pattern": 8},
    )
    assert 0 <= rb["composite_score"] <= 100
    print(f"✓ scoring: composite={rb['composite_score']:.1f}  "
          f"sub={ {k: round(v,1) for k,v in rb['sub_scores'].items()} }")


def test_cooldown():
    from cooldown import CooldownGuard
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        path = tf.name
    cd = CooldownGuard(path, default_hours=6.0)
    assert cd.is_cool("TESTUSDT")
    cd.mark("TESTUSDT")
    assert not cd.is_cool("TESTUSDT")
    assert cd.age_hours("TESTUSDT") < 0.01
    os.remove(path)
    print("✓ cooldown: marks and gates correctly")


def test_decision():
    from decision import decide
    d = decide(composite=80, ml_prob=0.7, btc={"state": "BULLISH",
                                                "score_modifier": 1.0,
                                                "freeze": False},
               quality_ok=True, cooldown_ok=True)
    assert d["decision"] == "APPROVED"
    d = decide(composite=45, ml_prob=0.3, btc={"state": "BULLISH",
                                                "score_modifier": 1.0,
                                                "freeze": False},
               quality_ok=True, cooldown_ok=True)
    assert d["decision"] == "REJECTED"
    d = decide(composite=65, ml_prob=0.4, btc={"state": "BULLISH",
                                                "score_modifier": 1.0,
                                                "freeze": False},
               quality_ok=True, cooldown_ok=True)
    assert d["decision"] == "WATCHLIST"
    print("✓ decision: APPROVED / WATCHLIST / REJECTED branches work")


def test_ml_engine():
    from ml_engine import PumpHunterML, append_signal_history, \
        update_signal_outcome, soft_adapt_weights
    from features import FEATURE_NAMES
    import tempfile

    test_path = os.path.join(tempfile.gettempdir(), "test_history.csv")
    model_path = os.path.join(tempfile.gettempdir(), "test_model.joblib")
    if os.path.exists(test_path):
        os.remove(test_path)
    if os.path.exists(model_path):
        os.remove(model_path)

    # Append 60 fake signals with random features and labels
    rng = np.random.default_rng(0)
    n = 60
    for i in range(n):
        feats = {f: float(rng.normal(0, 1)) for f in FEATURE_NAMES}
        feats["rvol"] = abs(feats["rvol"]) + 0.5
        feats["rsi_value"] = 50 + feats["rsi_value"] * 10
        row = {
            "timestamp": f"2024-01-01T00:{i:02d}:00+00:00",
            "symbol": f"FAKE{i}USDT",
            "label": np.nan,
            "score": 70 + rng.normal(0, 5),
            "ml_prob": 0.5,
            "outcome_score": np.nan,
            "composite_score": 70,
            "entry_price": 1.0,
        }
        row.update(feats)
        append_signal_history(test_path, row)
        # Mark first 30 as success, last 30 as fail
        if i < 30:
            update_signal_outcome(test_path, f"FAKE{i}USDT",
                                   row["timestamp"], 75.0, 1)
        else:
            update_signal_outcome(test_path, f"FAKE{i}USDT",
                                   row["timestamp"], 30.0, 0)

    ml = PumpHunterML(model_path, min_train=30)
    assert ml.has_enough_data(test_path)
    ok = ml.train(test_path)
    assert ok
    prob = ml.predict_proba(
        {f: float(rng.normal(0, 1)) for f in FEATURE_NAMES}
    )
    assert 0.0 <= prob <= 1.0
    base = {"technical": 12, "momentum": 12, "volume": 18, "vwap": 8,
            "atr_bb": 6, "structure": 10, "candle": 8, "mtf": 8, "pattern": 8}
    adapted = soft_adapt_weights(base, ml)
    s = sum(adapted.values())
    assert abs(s - 100) < 0.1, f"adapted weights sum to {s}"
    print(f"✓ ml_engine: train ok, prob={prob:.2f}, "
          f"adapted_weights_sum={s:.1f}")

    os.remove(test_path)
    if os.path.exists(model_path):
        os.remove(model_path)


def test_outcome():
    from outcome import compute_outcome_metrics
    rng = np.random.default_rng(1)
    # Build a 'forward' series that goes up 5% then back down
    n = 4
    closes = np.array([100, 103, 105, 102])
    highs = closes + 1
    lows = closes - 1
    opens = closes - 0.5
    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": [1000] * n,
    })
    m = compute_outcome_metrics(df, entry_price=100.0, forward_bars=3)
    assert m["peak_profit_pct"] > 0
    assert m["label"] in (0, 1)
    print(f"✓ outcome: peak={m['peak_profit_pct']:.1f}%  "
          f"dd={m['max_drawdown_pct']:.1f}%  "
          f"final={m['final_return_pct']:.1f}%  "
          f"score={m['outcome_score']:.1f}  label={m['label']}")


def test_dashboard():
    from dashboard import render_dashboard
    fake = {
        "timestamp": "2024-01-01T00:00:00+00:00",
        "weights": {"technical": 12, "momentum": 12, "volume": 18,
                    "vwap": 8, "atr_bb": 6, "structure": 10, "candle": 8,
                    "mtf": 8, "pattern": 8},
        "threshold": 75,
        "scanned": 2, "alerts_count": 1, "watchlist_count": 1,
        "btc": {"state": "BULLISH"},
        "alerts": [{
            "symbol": "TESTUSDT", "composite_score": 88.0, "ml_prob": 0.78,
            "decision": "APPROVED", "market_cap_usd": 5e6,
            "quote_volume_24h": 3e6, "rsi_value": 30, "rvol": 2.5,
        }],
        "watchlist": [{
            "symbol": "FOOUSDT", "composite_score": 65.0, "ml_prob": 0.45,
            "decision": "WATCHLIST", "market_cap_usd": 8e6,
            "quote_volume_24h": 2e6, "rsi_value": 50, "rvol": 1.3,
        }],
        "results": [
            {"symbol": "TESTUSDT", "composite_score": 88.0, "ml_prob": 0.78,
             "decision": "APPROVED", "market_cap_usd": 5e6,
             "quote_volume_24h": 3e6, "rsi_value": 30, "rvol": 2.5,
             "macd_hist": 0.01, "ema_alignment": "bullish", "patterns": []},
            {"symbol": "FOOUSDT", "composite_score": 65.0, "ml_prob": 0.45,
             "decision": "WATCHLIST", "market_cap_usd": 8e6,
             "quote_volume_24h": 2e6, "rsi_value": 50, "rvol": 1.3,
             "macd_hist": 0.001, "ema_alignment": "mixed", "patterns": []},
        ],
    }
    import json
    os.makedirs("data", exist_ok=True)
    with open("data/last_scan.json", "w", encoding="utf-8") as f:
        json.dump(fake, f, ensure_ascii=False, indent=2)
    p = render_dashboard()
    print(f"✓ dashboard: {p} ({os.path.getsize(p)} bytes)")


if __name__ == "__main__":
    print("=" * 60)
    print("PumpHunter-AI smoke tests")
    print("=" * 60)
    test_data_quality()
    test_indicators()
    test_structure()
    test_candle_quality()
    test_btc_filter()
    test_features()
    test_scoring()
    test_cooldown()
    test_decision()
    test_ml_engine()
    test_outcome()
    test_dashboard()
    print("=" * 60)
    print("✅ All smoke tests passed.")
