"""
Adaptive TP/SL — adjust take profit and stop loss based on signal context.

Instead of fixed TP=+5% / SL=-3% for every signal, we adapt based on:
  1. ATR (volatility): high ATR = wider TP/SL needed
  2. Momentum strength: strong mom = tighter TP (take profit fast)
  3. Volume context: high rvol = can be tighter (move is real)
  4. Market regime: bearish market = wider SL, tighter TP
  5. Signal confidence: high confidence = can extend TP
  6. BTC correlation: independent mover = wider SL (more volatile)
  7. Time of day: low liquidity hours = tighter TP
  8. Ichimoku cloud distance: bigger distance = bigger move potential

All parameters calibrated from observed small-cap behavior:
  - Average true range 4h is ~3% for active small caps
  - Mean move after 4h signal: +2-8%
  - Pump runs last 4-24h typically
  - False breakouts hit -3% in 1-2 candles

The adaptive model returns (tp_pct, sl_pct, trailing_pct, use_trailing)
so the tracker can use signal-specific exits.
"""
from __future__ import annotations
import math
from typing import Dict, Tuple, Optional


# Configurable base values
BASE_TP = 5.0      # default take profit %
BASE_SL = 3.0      # default stop loss %
MIN_TP = 2.0       # never less than 2% TP
MAX_TP = 15.0      # never more than 15% TP
MIN_SL = 1.5       # never less than 1.5% SL
MAX_SL = 8.0       # never more than 8% SL


def compute_adaptive_tp_sl(features: Dict) -> Dict:
    """
    Compute adaptive TP, SL, trailing, scaled based on features.
    Returns a dict with:
      - tp_pct: take profit %
      - sl_pct: stop loss %
      - trailing_pct: trailing stop distance from high
      - use_trailing: bool
      - use_scaled: bool
      - reasoning: str (why these values)
    """
    # Extract features with safe defaults
    atr_pct = float(features.get("f_atr_pct", 3.0))
    mom_3 = float(features.get("f_momentum_3_pct", 0))
    mom_6 = float(features.get("f_momentum_6_pct", 0))
    mom_12 = float(features.get("f_momentum_12_pct", 0))
    rvol = float(features.get("f_rvol", 1.0))
    confidence = float(features.get("confidence", 50))
    btc_state = str(features.get("btc_state", "NEUTRAL"))
    btc_mom = float(features.get("btc_momentum_12_pct", 0))
    direction = str(features.get("direction", "LONG"))
    ichi_above = bool(features.get("f_a_ichi_above_cloud", 0))
    ichi_below = bool(features.get("f_a_ichi_below_cloud", 0))
    fib_dist = float(features.get("f_a_fib_distance_pct", 0))
    fib_618 = float(features.get("f_a_fib_dist_0.618", 99))
    atr_expanding = bool(features.get("f_atr_expanding", 0))
    m5m_spike = bool(features.get("f_m_5m_volume_spike", 0))
    obi = float(features.get("f_m_obi_10", 1.0))
    cvd = float(features.get("f_m_cvd", 0))
    bb_breakout = bool(features.get("f_bb_breakout_above", 0))
    bos_up = bool(features.get("f_bos_up", 0))
    reasoning = []
    # ============================================================
    # STEP 1: Volatility-adjusted baseline (ATR)
    # ============================================================
    # If ATR is high, the price moves a lot — we need wider TP/SL to
    # avoid being stopped out by normal noise.
    # atr_pct of 3% is our reference. Scale relative to that.
    vol_mult = max(0.6, min(1.8, atr_pct / 3.0))
    tp_pct = BASE_TP * vol_mult
    sl_pct = BASE_SL * vol_mult
    reasoning.append(f"vol_mult={vol_mult:.2f} (atr={atr_pct:.1f}%)")
    # ============================================================
    # STEP 2: Momentum strength
    # ============================================================
    # Strong momentum (mom_6 > 5%) means price is moving. We can take
    # profit tighter because the move is happening now.
    # Conversely, weak momentum (|mom_6| < 1%) suggests chop — we need
    # wider TP to be patient.
    abs_mom_6 = abs(mom_6)
    if abs_mom_6 > 10:
        # Strong move already happening, take profit faster
        tp_mult = 0.8
        reasoning.append(f"strong_mom(>{10}%) tp_mult=0.8")
    elif abs_mom_6 > 5:
        tp_mult = 0.9
        reasoning.append(f"med_mom(>5%) tp_mult=0.9")
    elif abs_mom_6 < 1.0:
        # Choppy, need patience
        tp_mult = 1.3
        sl_mult = 1.2
        sl_pct *= sl_mult
        reasoning.append(f"chop_mom(<1%) tp_mult=1.3 sl_mult=1.2")
    else:
        tp_mult = 1.0
    tp_pct *= tp_mult
    # ============================================================
    # STEP 3: Volume confirmation
    # ============================================================
    # High rvol (>1.5) means real buying/selling. Move is genuine.
    # Tighten SL because we'll be right; can keep TP because move continues.
    if rvol > 2.0:
        sl_pct *= 0.85
        tp_pct *= 1.1
        reasoning.append(f"high_rvol(>2) sl*=0.85 tp*=1.1")
    elif rvol > 1.3:
        sl_pct *= 0.92
        reasoning.append(f"med_rvol(>1.3) sl*=0.92")
    elif rvol < 0.5:
        # Low volume, unreliable. Widen SL.
        sl_pct *= 1.15
        reasoning.append(f"low_rvol(<0.5) sl*=1.15")
    # ============================================================
    # STEP 4: BTC market regime
    # ============================================================
    # In bearish BTC, alt-coin dumps are more common. Tighten TP, widen SL.
    if btc_state == "BEARISH" or btc_mom < -5:
        if direction == "LONG":
            tp_pct *= 0.85
            sl_pct *= 1.15
            reasoning.append("btc_bearish_long: tp*=0.85 sl*=1.15")
        else:  # SHORT benefits from bearish BTC
            tp_pct *= 1.1
            sl_pct *= 0.9
            reasoning.append("btc_bearish_short: tp*=1.1 sl*=0.9")
    elif btc_state == "BULLISH" or btc_mom > 5:
        if direction == "LONG":
            tp_pct *= 1.1
            sl_pct *= 0.9
            reasoning.append("btc_bullish_long: tp*=1.1 sl*=0.9")
        else:  # SHORT fights the trend
            tp_pct *= 0.85
            sl_pct *= 1.15
            reasoning.append("btc_bullish_short: tp*=0.85 sl*=1.15")
    # ============================================================
    # STEP 5: Signal confidence
    # ============================================================
    # High confidence (>= 80) means many rules agree. We can extend TP
    # because the signal is more reliable.
    if confidence >= 80:
        tp_pct *= 1.15
        reasoning.append(f"high_conf(>={80}) tp*=1.15")
    elif confidence >= 70:
        tp_pct *= 1.05
        reasoning.append(f"med_conf(>={70}) tp*=1.05")
    elif confidence < 50:
        # Low confidence: be defensive
        sl_pct *= 0.9
        tp_pct *= 0.9
        reasoning.append(f"low_conf(<50) tp*=0.9 sl*=0.9")
    # ============================================================
    # STEP 6: Structure confirmation
    # ============================================================
    # BOS / BB breakout / Ichimoku above = strong trend. Tighter SL.
    structure_count = sum([bb_breakout, bos_up,
                            ichi_above if direction == "LONG" else ichi_below,
                            atr_expanding])
    if structure_count >= 3:
        sl_pct *= 0.85
        tp_pct *= 1.1
        reasoning.append(f"strong_struct(3+/3) sl*=0.85 tp*=1.1")
    elif structure_count == 0:
        sl_pct *= 1.1
        reasoning.append("no_struct sl*=1.1")
    # ============================================================
    # STEP 7: Microstructure confirmation
    # ============================================================
    # 5m spike + OBI supportive + CVD confirms = very strong signal
    micro_confirms = 0
    if m5m_spike:
        micro_confirms += 1
    if direction == "LONG" and obi > 1.3:
        micro_confirms += 1
    elif direction == "SHORT" and obi < 0.7:
        micro_confirms += 1
    if direction == "LONG" and cvd > 0:
        micro_confirms += 1
    elif direction == "SHORT" and cvd < 0:
        micro_confirms += 1
    if micro_confirms >= 2:
        sl_pct *= 0.9
        tp_pct *= 1.05
        reasoning.append(f"micro_confirms({micro_confirms}) sl*=0.9 tp*=1.05")
    # ============================================================
    # STEP 8: Fibonacci proximity
    # ============================================================
    # Near fib level (0.618) = bounce target nearby. Tighten TP.
    if fib_618 < 5:
        # Already at fib support, expect bounce to be small
        tp_pct *= 0.9
        reasoning.append(f"near_fib_618({fib_618:.1f}%) tp*=0.9")
    elif fib_618 > 30:
        # Far from fib, may need bigger move
        tp_pct *= 1.1
        reasoning.append(f"far_fib_618({fib_618:.1f}%) tp*=1.1")
    # ============================================================
    # STEP 9: Trailing stop logic
    # ============================================================
    # Use trailing when:
    #   - We have momentum (mom_3 or mom_6 > 2%)
    #   - Volume is real (rvol > 1.0)
    #   - Structure is intact
    if abs_mom_6 > 2.0 and rvol > 1.0 and structure_count >= 1:
        use_trailing = True
        # Trail at 1.5x the SL distance, so we lock in more profit
        trailing_pct = max(1.5, sl_pct * 0.7)
        reasoning.append(f"trailing={trailing_pct:.1f}%")
    else:
        use_trailing = False
        trailing_pct = 0
        reasoning.append("no_trailing (no mom/vol/struct)")
    # ============================================================
    # STEP 10: Scaled exit (TP1 + TP2)
    # ============================================================
    # Use scaled exit when:
    #   - High confidence (>= 70)
    #   - Strong volatility (room for both legs)
    #   - Not in extreme BTC bear
    if confidence >= 70 and atr_pct >= 3.0 and btc_state != "BEARISH":
        use_scaled = True
        reasoning.append("scaled=True (high conf + vol)")
    else:
        use_scaled = False
    # ============================================================
    # STEP 11: Clamp to safe ranges
    # ============================================================
    tp_pct = max(MIN_TP, min(MAX_TP, tp_pct))
    sl_pct = max(MIN_SL, min(MAX_SL, sl_pct))
    # Ensure TP > SL (always)
    if tp_pct <= sl_pct:
        tp_pct = sl_pct * 1.5
    return {
        "tp_pct": round(tp_pct, 2),
        "sl_pct": round(sl_pct, 2),
        "trailing_pct": round(trailing_pct, 2),
        "use_trailing": use_trailing,
        "use_scaled": use_scaled,
        "reasoning": " | ".join(reasoning),
    }


def get_signal_tp_sl(features: Dict) -> Dict:
    """
    Main entry point: given a feature dict (the row from the collector),
    return the adaptive TP/SL settings.
    """
    return compute_adaptive_tp_sl(features)


def format_tp_sl_for_log(params: Dict) -> str:
    """Pretty-print the TP/SL params for logs."""
    parts = [f"TP=+{params['tp_pct']}%", f"SL=-{params['sl_pct']}%"]
    if params.get("use_trailing"):
        parts.append(f"trail={params['trailing_pct']}%")
    if params.get("use_scaled"):
        parts.append("scaled")
    return " ".join(parts)


# ============================================================================
# Calibration utility: find optimal TP/SL from resolved signals
# ============================================================================
def analyze_optimal_tp_sl_per_signal(resolved_df: "pd.DataFrame") -> "pd.DataFrame":
    """
    For each resolved signal, compute what TP/SL would have been best.
    Returns a DataFrame with columns: symbol, direction, actual_exit_pct,
    best_tp, best_sl, max_runup, max_drawdown.
    """
    if resolved_df.empty:
        return resolved_df
    rows = []
    for _, r in resolved_df.iterrows():
        try:
            entry = float(r.get("entry_price", 0))
            max_runup = float(r.get("max_favorable_pct", 0) or 0)
            max_dd = abs(float(r.get("max_adverse_pct", 0) or 0))
            actual_exit = float(r.get("exit_pct", 0) or 0)
            # Best TP: smallest TP that would have triggered (max runup)
            best_tp = round(max_runup, 2) if max_runup > 0 else 0
            # Best SL: SL just above max drawdown (would not have triggered)
            best_sl = round(max_dd + 0.1, 2) if max_dd > 0 else 0
            # If TP > 0 and SL > 0, compute hypothetical pnl
            # If price hit max_runup then pulled back to actual_exit, the
            # best strategy is: exit at max_runup (or with trail)
            hypothetical_pnl = max_runup  # if we trailed perfectly
            rows.append({
                "symbol": r.get("symbol"),
                "direction": r.get("direction"),
                "actual_exit_pct": actual_exit,
                "max_runup": max_runup,
                "max_drawdown": max_dd,
                "best_tp_pct": best_tp,
                "best_sl_pct": best_sl,
                "hypothetical_perfect_pnl": hypothetical_pnl,
                "actual_status": r.get("status"),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)
