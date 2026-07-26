"""
Ultra-strict v2: enhanced with extended indicators.

Adds 8 new top-ranked extended indicators from backtest analysis:
  1. ex_ulcer_score    - downside volatility (vol)
  2. ex_cmf            - accumulation/distribution (vol)
  3. ex_ulcer          - ulcer index value
  4. ex_uo             - ultimate oscillator (mom)
  5. ex_obv_norm       - OBV slope normalized (vol)
  6. ex_stoch_k        - stochastic %K (mom)
  7. ex_williams_r     - Williams %R (mom)
  8. ex_tsi_score      - TSI score (mom)

Each adds a "score boost" (0-15 points) to the confidence,
or "score penalty" (-15 points) for bearish conditions.

Final: conf_new = conf + ex_score_boost
       if conf_new < threshold: REJECT
"""
from __future__ import annotations
from typing import Dict, Tuple
import pandas as pd


def compute_extended_score(features: Dict, direction: str) -> Tuple[float, list]:
    """
    Compute score boost/penalty from extended indicators + adaptive weights.

    Returns:
      (score_delta, reasons)
      score_delta: -100 to +50, added to base confidence
      reasons: list of "boost:..." or "penalty:..." strings
    """
    score = 0.0
    reasons = []
    # === NEW: Adaptive learned weights from self-training ===
    try:
        from .adaptive_filter import load_feature_weights
        weights = load_feature_weights()
        for feat, w in weights.items():
            v = features.get(feat, features.get(f"f_{feat}", 0))
            if v is None:
                continue
            try:
                v_float = float(v)
            except Exception:
                continue
            # Sign-aware
            if direction == "LONG":
                contrib = v_float * w / 10.0
            else:
                contrib = -v_float * w / 10.0
            # Clip each contribution
            contrib = max(-5, min(5, contrib))
            score += contrib
            if abs(contrib) > 2:
                sign = "+" if contrib > 0 else "-"
                reasons.append(f"adapt_{sign}{feat.split('_')[-1]}")
    except Exception:
        pass
    # 1. Ulcer Index score (volatility)
    ulcer_s = float(features.get("f_ex_ulcer_score", 50))
    # Low ulcer_score = low downside vol = good for entry
    if ulcer_s >= 70:  # very low downside vol → great entry
        score += 12
        reasons.append("boost:ulcer_low_vol")
    elif ulcer_s >= 60:
        score += 6
    elif ulcer_s <= 30:  # high downside vol → risky
        score -= 10
        reasons.append("penalty:ulcer_high_vol")
    # 2. CMF (Chaikin Money Flow) — accumulation/distribution
    cmf = float(features.get("f_ex_cmf", 0))
    cmf_pos = bool(features.get("f_ex_cmf_positive", 0))
    if direction == "LONG":
        if cmf > 0.1:  # strong accumulation
            score += 10
            reasons.append("boost:cmf_accum_long")
        elif cmf > 0:
            score += 5
        elif cmf < -0.1:  # distribution
            score -= 12
            reasons.append("penalty:cmf_dist_long")
    else:  # SHORT
        if cmf < -0.1:
            score += 10
            reasons.append("boost:cmf_dist_short")
        elif cmf < 0:
            score += 5
        elif cmf > 0.1:
            score -= 12
            reasons.append("penalty:cmf_accum_short")
    # 3. Ultimate Oscillator (UO)
    uo = float(features.get("f_ex_uo", 50))
    if direction == "LONG":
        if uo < 30:  # oversold → bounce likely
            score += 8
            reasons.append("boost:uo_oversold_long")
        elif uo > 70:  # overbought → avoid
            score -= 8
            reasons.append("penalty:uo_overbought_long")
    else:  # SHORT
        if uo > 70:
            score += 8
            reasons.append("boost:uo_overbought_short")
        elif uo < 30:
            score -= 8
            reasons.append("penalty:uo_oversold_short")
    # 4. OBV normalized (volume momentum)
    obv_slope = float(features.get("f_ex_obv_slope_norm", 0))
    if direction == "LONG":
        if obv_slope > 0.05:  # strong positive volume
            score += 8
            reasons.append("boost:obv_vol_long")
        elif obv_slope < -0.05:
            score -= 6
            reasons.append("penalty:obv_vol_long")
    else:  # SHORT
        if obv_slope < -0.05:
            score += 8
            reasons.append("boost:obv_vol_short")
        elif obv_slope > 0.05:
            score -= 6
            reasons.append("penalty:obv_vol_short")
    # 5. Stochastic %K
    stoch_k = float(features.get("f_ex_stoch_k", 50))
    if direction == "LONG":
        if stoch_k < 20:  # oversold
            score += 6
            reasons.append("boost:stoch_oversold_long")
        elif stoch_k > 80:  # overbought — risk
            score -= 8
            reasons.append("penalty:stoch_overbought_long")
    else:  # SHORT
        if stoch_k > 80:
            score += 6
            reasons.append("boost:stoch_overbought_short")
        elif stoch_k < 20:
            score -= 8
            reasons.append("penalty:stoch_oversold_short")
    # 6. Williams %R
    wr = float(features.get("f_ex_williams_r", -50))
    if direction == "LONG":
        if wr < -80:  # oversold
            score += 5
            reasons.append("boost:williams_oversold")
        elif wr > -20:  # overbought
            score -= 5
            reasons.append("penalty:williams_overbought")
    else:  # SHORT
        if wr > -20:
            score += 5
            reasons.append("boost:williams_overbought")
        elif wr < -80:
            score -= 5
            reasons.append("penalty:williams_oversold")
    # 7. TSI score
    tsi_s = float(features.get("f_ex_tsi_score", 50))
    if direction == "LONG":
        if tsi_s >= 60:  # positive momentum
            score += 5
        elif tsi_s <= 40:
            score -= 5
    else:  # SHORT
        if tsi_s <= 40:  # negative momentum
            score += 5
        elif tsi_s >= 60:
            score -= 5
    # 8. ROC score
    roc_s = float(features.get("f_ex_roc_score", 50))
    if direction == "LONG" and roc_s >= 60:
        score += 3
    elif direction == "SHORT" and roc_s <= 40:
        score += 3
    return score, reasons


def is_ultra_setup_v2(features: Dict, direction: str) -> Tuple[bool, str, float]:
    """
    Ultra-strict v2: base filters + extended indicators.

    Returns:
      (passes, reason_if_fail, adjusted_confidence)
    """
    # Layer 0: anti-late filter (must pass first)
    try:
        from .anti_late import check_anti_late
        anti_feats = {
            "momentum_1_pct": float(features.get("f_momentum_1_pct", 0)),
            "momentum_3_pct": float(features.get("f_momentum_3_pct", 0)),
            "momentum_6_pct": float(features.get("f_momentum_6_pct", 0)),
            "momentum_acceleration": float(features.get("f_momentum_acceleration", 0)),
            "candle_strength": float(features.get("f_candle_strength", 0.5)),
            "big_wick_top": bool(features.get("f_big_wick_top", False)),
            "wick_body_ratio": float(features.get("f_wick_body_ratio", 0.0)),
        }
        ok_late, pen, late_reason = check_anti_late(anti_feats, direction)
        if not ok_late:
            return False, f"anti_late_{late_reason}", 0.0
    except Exception:
        pass

    # === Base filters (same as v1) ===
    atr = float(features.get("f_atr_pct", 0))
    mom_3 = float(features.get("f_momentum_3_pct", 0))
    mom_6 = float(features.get("f_momentum_6_pct", 0))
    confidence = float(features.get("confidence", 0))
    ichi_above = bool(features.get("f_a_ichi_above_cloud", 0))
    ichi_below = bool(features.get("f_a_ichi_below_cloud", 0))
    bb_breakout = bool(features.get("f_bb_breakout_above", 0))
    bos_up = bool(features.get("f_bos_up", 0))
    atr_exp = bool(features.get("f_atr_expanding", 0))
    m5m_spike = bool(features.get("f_m_5m_volume_spike", 0))
    obi = float(features.get("f_m_obi_10", 1))
    cvd = float(features.get("f_m_cvd", 0))
    btc_mom = float(features.get("btc_momentum_12_pct", 0))
    score_long = float(features.get("score_long", 0))
    score_short = float(features.get("score_short", 0))
    mom_12 = float(features.get("f_momentum_12_pct", 0))

    fails = []
    if atr < 3.0 or atr > 8.0:
        fails.append(f"atr({atr:.1f})")
    if direction == "LONG" and mom_3 >= 8:
        fails.append(f"overext_long(m3={mom_3:.1f})")
    if direction == "SHORT" and mom_3 <= -8:
        fails.append(f"overext_short(m3={mom_3:.1f})")
    if abs(mom_6) < 1.5:
        fails.append(f"chop(m6={mom_6:.1f})")
    if direction == "LONG" and btc_mom < -2:
        fails.append("btc_against")
    if direction == "SHORT" and btc_mom > 2:
        fails.append("btc_against")
    if fails:
        return False, ",".join(fails), 0.0

    # Structure (need 4/5)
    struct_count = 0
    if direction == "LONG":
        if ichi_above: struct_count += 1
        if bb_breakout: struct_count += 1
        if bos_up: struct_count += 1
        if atr_exp: struct_count += 1
        if mom_6 > 0 and mom_12 > 0: struct_count += 1
    else:
        if ichi_below: struct_count += 1
        if not bb_breakout: struct_count += 1
        if not bos_up: struct_count += 1
        if atr_exp: struct_count += 1
        if mom_6 < 0 and mom_12 < 0: struct_count += 1
    if struct_count < 4:
        return False, f"struct({struct_count}/5)", 0.0

    # Microstructure (need 1+)
    micro = 0
    if m5m_spike: micro += 1
    if direction == "LONG" and obi > 1.3: micro += 1
    elif direction == "SHORT" and obi < 0.7: micro += 1
    if direction == "LONG" and cvd > 0: micro += 1
    elif direction == "SHORT" and cvd < 0: micro += 1
    if micro < 1:
        return False, "no_micro", 0.0

    # Score threshold
    if direction == "LONG" and score_long < 50:
        return False, f"low_score({score_long:.0f})", 0.0
    if direction == "SHORT" and score_short < 50:
        return False, f"low_score({score_short:.0f})", 0.0

    # === NEW: Extended indicator score boost ===
    ex_delta, ex_reasons = compute_extended_score(features, direction)
    adjusted_conf = confidence + ex_delta
    # Final threshold: 60 (base) or higher if ex score is very positive
    threshold = 60
    if ex_delta >= 20:  # very strong extended signal
        threshold = 55
    if adjusted_conf < threshold:
        return False, f"ext_conf_low({adjusted_conf:.0f}<{threshold})", adjusted_conf
    return True, "", adjusted_conf


# Backward compatibility alias
def is_ultra_setup(features: Dict, direction: str) -> Tuple[bool, str]:
    """Alias for v2 (returns just bool/str)."""
    ok, reason, _ = is_ultra_setup_v2(features, direction)
    return ok, reason


def _get_feat(full_feats, row, key, default=0):
    """Get feature from features_json first, then row, then default."""
    if key in full_feats and full_feats[key] is not None:
        return full_feats[key]
    if key in row.index and pd.notna(row[key]):
        return row[key]
    return default


def get_ultra_picks_v2(min_confidence: float = 60.0) -> Dict:
    """
    Get today's ULTRA-STRICT V2 picks (with extended indicators).
    Expected: 0-3 picks per day.
    """
    import json
    from . import db as database
    features = database.get_features()
    if features.empty:
        return {"error": "no features"}
    features["ts"] = pd.to_datetime(features["ts"], utc=True, errors="coerce")
    last_ts = features["ts"].max()
    last = features[features["ts"] == last_ts].copy()
    if last.empty:
        return {"error": "no last cycle"}
    longs = []
    shorts = []
    for _, r in last.iterrows():
        direction = r.get("direction")
        if direction not in ("LONG", "SHORT"):
            continue
        if r.get("confidence", 0) < min_confidence:
            continue
        # Load full features from features_json
        fj = r.get("features_json", "{}")
        if isinstance(fj, str):
            try:
                full = json.loads(fj)
            except Exception:
                full = {}
        else:
            full = {}
        # Build features dict
        feats = {
            "f_atr_pct": _get_feat(full, r, "f_atr_pct", r.get("ind_atr_pct", 0)),
            "f_momentum_1_pct": _get_feat(full, r, "f_momentum_1_pct", 0),
            "f_momentum_3_pct": _get_feat(full, r, "f_momentum_3_pct",
                                          r.get("ind_momentum_3_pct", 0)),
            "f_momentum_6_pct": _get_feat(full, r, "f_momentum_6_pct",
                                          r.get("ind_momentum_6_pct", 0)),
            "f_momentum_12_pct": _get_feat(full, r, "f_momentum_12_pct", 0),
            "f_momentum_acceleration": _get_feat(full, r, "f_momentum_acceleration", 0),
            "f_a_ichi_above_cloud": _get_feat(full, r, "f_a_ichi_above_cloud", 0),
            "f_a_ichi_below_cloud": _get_feat(full, r, "f_a_ichi_below_cloud", 0),
            "f_bb_breakout_above": _get_feat(full, r, "f_bb_breakout_above", 0),
            "f_bos_up": _get_feat(full, r, "f_bos_up", 0),
            "f_atr_expanding": _get_feat(full, r, "f_atr_expanding", 0),
            "f_m_5m_volume_spike": _get_feat(full, r, "f_m_5m_volume_spike", 0),
            "f_m_obi_10": _get_feat(full, r, "f_m_obi_10", 1),
            "f_m_cvd": _get_feat(full, r, "f_m_cvd", 0),
            "f_a_wave_position": _get_feat(full, r, "f_a_wave_position", ""),
            "f_candle_strength": _get_feat(full, r, "f_candle_strength", 0.5),
            "f_big_wick_top": _get_feat(full, r, "f_big_wick_top", 0),
            "f_wick_body_ratio": _get_feat(full, r, "f_wick_body_ratio", 0),
            # Extended indicators
            "f_ex_ulcer_score": _get_feat(full, r, "f_ex_ulcer_score", 50),
            "f_ex_cmf": _get_feat(full, r, "f_ex_cmf", 0),
            "f_ex_cmf_positive": _get_feat(full, r, "f_ex_cmf_positive", 0),
            "f_ex_uo": _get_feat(full, r, "f_ex_uo", 50),
            "f_ex_obv_slope_norm": _get_feat(full, r, "f_ex_obv_slope_norm", 0),
            "f_ex_stoch_k": _get_feat(full, r, "f_ex_stoch_k", 50),
            "f_ex_williams_r": _get_feat(full, r, "f_ex_williams_r", -50),
            "f_ex_tsi_score": _get_feat(full, r, "f_ex_tsi_score", 50),
            "f_ex_roc_score": _get_feat(full, r, "f_ex_roc_score", 50),
            "confidence": r.get("confidence", 0),
            "btc_momentum_12_pct": r.get("btc_momentum_12_pct", 0),
            "score_long": r.get("score_long", 0),
            "score_short": r.get("score_short", 0),
        }
        ok, reason, adj_conf = is_ultra_setup_v2(feats, direction)
        if ok:
            if direction == "LONG":
                longs.append((r, feats, adj_conf))
            else:
                shorts.append((r, feats, adj_conf))
    return {
        "timestamp": str(last_ts),
        "longs": longs,
        "shorts": shorts,
    }


# Self-test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/toobit-scanner")
    from src.toobit_client import ToobitClient
    from src.extended_indicators import compute_all_extended
    import time
    tb = ToobitClient()
    print("Testing ultra_v2 on 5 symbols...")
    for sym in ["ALICEUSDT", "TLMUSDT", "HYPERUSDT", "WCTUSDT", "RECALLUSDT"]:
        df = tb.get_klines(sym, "4h", 100)
        if df.empty or len(df) < 60:
            continue
        ex = compute_all_extended(df)
        feats = {
            "f_atr_pct": 5.0,
            "f_momentum_3_pct": 2.0,
            "f_momentum_6_pct": 3.0,
            "f_momentum_12_pct": 4.0,
            "f_a_ichi_above_cloud": 1,
            "f_bb_breakout_above": 1,
            "f_bos_up": 1,
            "f_atr_expanding": 1,
            "f_m_5m_volume_spike": 1,
            "f_m_obi_10": 1.5,
            "f_m_cvd": 100,
            "confidence": 65,
            "btc_momentum_12_pct": 3.0,
            "score_long": 55,
            "score_short": 30,
        }
        for k, v in ex.items():
            feats[f"f_{k}"] = v
        ok, reason, adj_conf = is_ultra_setup_v2(feats, "LONG")
        delta, reasons = compute_extended_score(feats, "LONG")
        print(f"\n{sym} (LONG):")
        print(f"  Base conf: {feats['confidence']}")
        print(f"  Extended score delta: {delta:+.1f}")
        print(f"  Adjusted conf: {adj_conf:.1f}")
        print(f"  Pass: {ok}, Reason: {reason}")
        print(f"  Reasons: {reasons[:5]}")
        time.sleep(0.2)
