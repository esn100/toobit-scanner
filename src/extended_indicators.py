"""
Extended technical indicators for PumpHunter-AI (v2).

All-new indicators that complement the existing system.
Each function returns a flat dict of features with `ex_` prefix
to avoid collision with existing features (f_/a_/m_/s_/ind_).

CATEGORIES (8 new indicator groups):

  MOMENTUM OSCILLATORS:
    - Stochastic Oscillator (%K, %D, crossover)
    - Stochastic RSI (smoother, less lag)
    - Williams %R (inverted stochastic)
    - CCI (commodity channel index)
    - Awesome Oscillator (momentum histogram)
    - TSI (true strength index)
    - Ultimate Oscillator (3-timeframe composite)
    - ROC (rate of change)

  TREND STRENGTH:
    - ADX (average directional index)
    - Plus DI / Minus DI
    - Aroon (up/down + oscillator)
    - Parabolic SAR
    - Vortex (VI+ / VI-)
    - SuperTrend
    - KST (know sure thing)
    - TRIX (triple-smoothed momentum)

  VOLUME-BASED:
    - MFI (money flow index) — RSI + volume
    - CMF (chaikin money flow) — accumulation/distribution
    - OBV (on balance volume) — cumulative
    - A/D (accumulation/distribution) — smart money
    - Force Index — price × volume
    - EOM (ease of movement) — vol/price change
    - VPT (volume price trend) — OBV smoothed

  VOLATILITY:
    - Keltner Channels — ATR-based envelope
    - Donchian Channels — high/low breakout
    - Ulcer Index — downside volatility
    - Mass Index — reversal detection
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict
import ta


# ===========================================================================
# MOMENTUM OSCILLATORS
# ===========================================================================
def stochastic_features(df: pd.DataFrame) -> dict:
    """Stochastic Oscillator: %K, %D, crossover signal."""
    out = {"ex_stoch_k": 50.0, "ex_stoch_d": 50.0,
           "ex_stoch_crossover": 0, "ex_stoch_overbought": 0,
           "ex_stoch_oversold": 0, "ex_stoch_score": 50.0}
    try:
        stoch = ta.momentum.StochasticOscillator(
            df["high"], df["low"], df["close"], window=14, smooth_window=3
        )
        k = float(stoch.stoch().iloc[-1])
        d = float(stoch.stoch_signal().iloc[-1])
        out["ex_stoch_k"] = k
        out["ex_stoch_d"] = d
        out["ex_stoch_crossover"] = int(k > d)  # bullish if K > D
        out["ex_stoch_overbought"] = int(k > 80)
        out["ex_stoch_oversold"] = int(k < 20)
        # Score: 0-100
        # - Oversold (k<20) = buy signal for LONG
        # - Overbought (k>80) = sell signal
        # - Crossover bullish = good
        score = 50.0
        if k < 20: score += 20  # oversold = potential bounce
        elif k > 80: score -= 20  # overbought = potential drop
        if k > d: score += 5
        else: score -= 5
        # Smoothed score in middle range
        out["ex_stoch_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def stochrsi_features(df: pd.DataFrame) -> dict:
    """Stochastic RSI: smoother than RSI, less lag."""
    out = {"ex_stochrsi_k": 50.0, "ex_stochrsi_d": 50.0,
           "ex_stochrsi_crossover": 0, "ex_stochrsi_score": 50.0}
    try:
        sr = ta.momentum.StochRSIIndicator(
            df["close"], window=14, smooth1=3, smooth2=3
        )
        k = float(sr.stochrsi().iloc[-1]) * 100
        d = float(sr.stochrsi_d().iloc[-1]) * 100
        out["ex_stochrsi_k"] = k
        out["ex_stochrsi_d"] = d
        out["ex_stochrsi_crossover"] = int(k > d)
        score = 50.0
        if k < 20: score += 25  # very oversold
        elif k > 80: score -= 25
        if k > d: score += 10
        else: score -= 10
        out["ex_stochrsi_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def williams_r_features(df: pd.DataFrame) -> dict:
    """Williams %R: inverted stochastic (range -100 to 0)."""
    out = {"ex_williams_r": -50.0, "ex_williams_oversold": 0,
           "ex_williams_overbought": 0, "ex_williams_score": 50.0}
    try:
        wr = ta.momentum.WilliamsRIndicator(
            df["high"], df["low"], df["close"], lbp=14
        )
        v = float(wr.williams_r().iloc[-1])
        out["ex_williams_r"] = v
        out["ex_williams_oversold"] = int(v < -80)  # buy signal
        out["ex_williams_overbought"] = int(v > -20)
        score = 50.0
        if v < -80: score += 20
        elif v > -20: score -= 20
        out["ex_williams_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def cci_features(df: pd.DataFrame) -> dict:
    """CCI: commodity channel index, momentum + volatility."""
    out = {"ex_cci": 0.0, "ex_cci_oversold": 0, "ex_cci_overbought": 0,
           "ex_cci_score": 50.0}
    try:
        cci = ta.trend.CCIIndicator(
            df["high"], df["low"], df["close"], window=20
        )
        v = float(cci.cci().iloc[-1])
        out["ex_cci"] = v
        out["ex_cci_oversold"] = int(v < -100)
        out["ex_cci_overbought"] = int(v > 100)
        score = 50.0
        if v < -200: score += 20  # extreme oversold
        elif v < -100: score += 10
        elif v > 200: score -= 20
        elif v > 100: score -= 10
        out["ex_cci_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def ao_features(df: pd.DataFrame) -> dict:
    """Awesome Oscillator: momentum histogram (5 vs 34 SMA)."""
    out = {"ex_ao": 0.0, "ex_ao_positive": 0, "ex_ao_saucer": 0,
           "ex_ao_score": 50.0}
    try:
        ao = ta.momentum.AwesomeOscillatorIndicator(
            df["high"], df["low"], window1=5, window2=34
        )
        v = float(ao.awesome_oscillator().iloc[-1])
        v_prev = float(ao.awesome_oscillator().iloc[-2]) if len(df) > 1 else 0
        out["ex_ao"] = v
        out["ex_ao_positive"] = int(v > 0)
        # Saucer: 3 consecutive bars above zero, second < first, third > second
        ao_series = ao.awesome_oscillator()
        if len(ao_series) >= 3:
            v3 = ao_series.iloc[-3:].values
            if v3[0] > 0 and v3[1] < v3[0] and v3[2] > v3[1]:
                out["ex_ao_saucer"] = 1
        score = 50.0
        if v > 0: score += 10
        if v > v_prev: score += 10  # momentum increasing
        if v < 0 and v > v_prev: score += 5  # turning from negative
        out["ex_ao_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def tsi_features(df: pd.DataFrame) -> dict:
    """TSI: True Strength Index, double-smoothed momentum."""
    out = {"ex_tsi": 0.0, "ex_tsi_positive": 0, "ex_tsi_score": 50.0}
    try:
        tsi = ta.momentum.TSIIndicator(df["close"], window_slow=25, window_fast=13)
        v = float(tsi.tsi().iloc[-1])
        out["ex_tsi"] = v
        out["ex_tsi_positive"] = int(v > 0)
        score = 50.0 + v  # -100 to +100 range typically
        out["ex_tsi_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def uo_features(df: pd.DataFrame) -> dict:
    """Ultimate Oscillator: 3-timeframe composite (7,14,28)."""
    out = {"ex_uo": 50.0, "ex_uo_oversold": 0, "ex_uo_score": 50.0}
    try:
        uo = ta.momentum.UltimateOscillator(
            df["high"], df["low"], df["close"],
            window1=7, window2=14, window3=28
        )
        v = float(uo.ultimate_oscillator().iloc[-1])
        out["ex_uo"] = v
        out["ex_uo_oversold"] = int(v < 30)
        score = 50.0
        if v < 30: score += 20
        elif v > 70: score -= 20
        out["ex_uo_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def roc_features(df: pd.DataFrame) -> dict:
    """ROC: Rate of Change (simple momentum)."""
    out = {"ex_roc": 0.0, "ex_roc_positive": 0, "ex_roc_score": 50.0}
    try:
        roc = ta.momentum.ROCIndicator(df["close"], window=12)
        v = float(roc.roc().iloc[-1])
        out["ex_roc"] = v
        out["ex_roc_positive"] = int(v > 0)
        score = 50.0 + max(-40, min(40, v * 4))
        out["ex_roc_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


# ===========================================================================
# TREND STRENGTH
# ===========================================================================
def adx_features(df: pd.DataFrame) -> dict:
    """ADX: Average Directional Index (trend STRENGTH, not direction)."""
    out = {"ex_adx": 0.0, "ex_adx_strong": 0, "ex_adx_weak": 0,
           "ex_plus_di": 0.0, "ex_minus_di": 0.0, "ex_adx_score": 50.0}
    try:
        adx = ta.trend.ADXIndicator(
            df["high"], df["low"], df["close"], window=14
        )
        adx_v = float(adx.adx().iloc[-1])
        plus_di = float(adx.adx_pos().iloc[-1])
        minus_di = float(adx.adx_neg().iloc[-1])
        out["ex_adx"] = adx_v
        out["ex_plus_di"] = plus_di
        out["ex_minus_di"] = minus_di
        out["ex_adx_strong"] = int(adx_v > 25)
        out["ex_adx_weak"] = int(adx_v < 20)
        score = 50.0
        if adx_v > 25: score += 15
        elif adx_v > 20: score += 5
        else: score -= 5
        # Direction: plus_di > minus_di = uptrend
        if plus_di > minus_di: score += 10
        else: score -= 10
        out["ex_adx_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def aroon_features(df: pd.DataFrame) -> dict:
    """Aroon: detects new trend starts (periods since high/low)."""
    out = {"ex_aroon_up": 50.0, "ex_aroon_down": 50.0, "ex_aroon_score": 50.0,
           "ex_aroon_crossover": 0}
    try:
        ar = ta.trend.AroonIndicator(df["high"], df["low"], window=25)
        up = float(ar.aroon_up().iloc[-1])
        down = float(ar.aroon_down().iloc[-1])
        out["ex_aroon_up"] = up
        out["ex_aroon_down"] = down
        out["ex_aroon_crossover"] = int(up > down)
        score = 50.0
        if up > 70: score += 15  # strong uptrend
        if down > 70: score -= 15
        if up > down: score += 5
        out["ex_aroon_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def psar_features(df: pd.DataFrame) -> dict:
    """Parabolic SAR: trailing stop + reversal signal (manual impl)."""
    out = {"ex_psar": 0.0, "ex_psar_above": 0, "ex_psar_below": 0,
           "ex_psar_score": 50.0}
    try:
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(closes)
        if n < 5:
            return out
        # Manual PSAR
        af_init = 0.02
        af_step = 0.02
        af_max = 0.2
        is_long = True
        af = af_init
        ep = highs[0]
        sar = lows[0]
        for i in range(1, n):
            sar = sar + af * (ep - sar)
            if is_long:
                sar = min(sar, lows[i-1], lows[max(0, i-2)])
                if lows[i] < sar:
                    is_long = False
                    sar = ep
                    ep = lows[i]
                    af = af_init
                else:
                    if highs[i] > ep:
                        ep = highs[i]
                        af = min(af + af_step, af_max)
            else:
                sar = max(sar, highs[i-1], highs[max(0, i-2)])
                if highs[i] > sar:
                    is_long = True
                    sar = ep
                    ep = highs[i]
                    af = af_init
                else:
                    if lows[i] < ep:
                        ep = lows[i]
                        af = min(af + af_step, af_max)
        psar_v = float(sar)
        close = float(closes[-1])
        out["ex_psar"] = psar_v
        out["ex_psar_above"] = int(psar_v > close)  # bearish
        out["ex_psar_below"] = int(psar_v < close)  # bullish
        out["ex_psar_bullish_trend"] = int(is_long)
        score = 50.0
        if is_long: score += 20
        else: score -= 20
        out["ex_psar_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def vortex_features(df: pd.DataFrame) -> dict:
    """Vortex: trend confirmation (VI+ vs VI-)."""
    out = {"ex_vortex_plus": 0.0, "ex_vortex_minus": 0.0, "ex_vortex_score": 50.0}
    try:
        vx = ta.trend.VortexIndicator(
            df["high"], df["low"], df["close"], window=14
        )
        plus = float(vx.vortex_indicator_pos().iloc[-1])
        minus = float(vx.vortex_indicator_neg().iloc[-1])
        out["ex_vortex_plus"] = plus
        out["ex_vortex_minus"] = minus
        score = 50.0
        if plus > minus: score += 15
        else: score -= 15
        if plus > 1.0: score += 5
        out["ex_vortex_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def supertrend_features(df: pd.DataFrame) -> dict:
    """SuperTrend: trend-following indicator."""
    out = {"ex_supertrend": 0.0, "ex_supertrend_bullish": 0,
           "ex_supertrend_distance_pct": 0.0, "ex_supertrend_score": 50.0}
    try:
        # Manual implementation
        period = 10
        multiplier = 3.0
        hl2 = (df["high"] + df["low"]) / 2
        atr = ta.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], window=period
        ).average_true_range()
        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr
        # Vectorised SuperTrend
        st = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        st.iloc[0] = lower.iloc[0]
        direction.iloc[0] = 1
        for i in range(1, len(df)):
            if df["close"].iloc[i] > upper.iloc[i - 1]:
                direction.iloc[i] = 1
            elif df["close"].iloc[i] < lower.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] == 1:
                st.iloc[i] = lower.iloc[i]
            else:
                st.iloc[i] = upper.iloc[i]
        last_st = float(st.iloc[-1])
        last_close = float(df["close"].iloc[-1])
        out["ex_supertrend"] = last_st
        out["ex_supertrend_bullish"] = int(direction.iloc[-1] == 1)
        out["ex_supertrend_distance_pct"] = float(
            (last_close - last_st) / max(last_st, 1e-12) * 100
        )
        score = 50.0
        if direction.iloc[-1] == 1: score += 20
        else: score -= 20
        out["ex_supertrend_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def kst_features(df: pd.DataFrame) -> dict:
    """KST: Know Sure Thing, momentum composite."""
    out = {"ex_kst": 0.0, "ex_kst_signal": 0.0, "ex_kst_crossover": 0,
           "ex_kst_score": 50.0}
    try:
        kst = ta.trend.KSTIndicator(
            df["close"], roc1=10, roc2=15, roc3=20, roc4=30,
            window1=10, window2=10, window3=10, window4=15,
            nsig=9
        )
        k = float(kst.kst().iloc[-1])
        s = float(kst.kst_sig().iloc[-1])
        out["ex_kst"] = k
        out["ex_kst_signal"] = s
        out["ex_kst_crossover"] = int(k > s)
        score = 50.0
        if k > s: score += 15
        else: score -= 15
        if k > 0: score += 5
        out["ex_kst_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def trix_features(df: pd.DataFrame) -> dict:
    """TRIX: triple-smoothed momentum, very low noise."""
    out = {"ex_trix": 0.0, "ex_trix_positive": 0, "ex_trix_score": 50.0}
    try:
        tr = ta.trend.TRIXIndicator(df["close"], window=15)
        v = float(tr.trix().iloc[-1])
        out["ex_trix"] = v
        out["ex_trix_positive"] = int(v > 0)
        score = 50.0 + max(-40, min(40, v * 100))
        out["ex_trix_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


# ===========================================================================
# VOLUME-BASED
# ===========================================================================
def mfi_features(df: pd.DataFrame) -> dict:
    """MFI: Money Flow Index, volume-weighted RSI."""
    out = {"ex_mfi": 50.0, "ex_mfi_oversold": 0, "ex_mfi_overbought": 0,
           "ex_mfi_score": 50.0}
    try:
        mfi = ta.volume.MFIIndicator(
            df["high"], df["low"], df["close"], df["volume"], window=14
        )
        v = float(mfi.money_flow_index().iloc[-1])
        out["ex_mfi"] = v
        out["ex_mfi_oversold"] = int(v < 20)
        out["ex_mfi_overbought"] = int(v > 80)
        score = 50.0
        if v < 20: score += 25
        elif v < 30: score += 10
        elif v > 80: score -= 25
        elif v > 70: score -= 10
        out["ex_mfi_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def cmf_features(df: pd.DataFrame) -> dict:
    """CMF: Chaikin Money Flow, accumulation/distribution."""
    out = {"ex_cmf": 0.0, "ex_cmf_positive": 0, "ex_cmf_score": 50.0}
    try:
        cmf = ta.volume.ChaikinMoneyFlowIndicator(
            df["high"], df["low"], df["close"], df["volume"], window=20
        )
        v = float(cmf.chaikin_money_flow().iloc[-1])
        out["ex_cmf"] = v
        out["ex_cmf_positive"] = int(v > 0)
        score = 50.0
        if v > 0.1: score += 20
        elif v > 0: score += 5
        elif v < -0.1: score -= 20
        elif v < 0: score -= 5
        out["ex_cmf_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def obv_features(df: pd.DataFrame) -> dict:
    """OBV: On Balance Volume (NORMALIZED)."""
    out = {"ex_obv_norm": 0.0, "ex_obv_slope_norm": 0.0,
           "ex_obv_score": 50.0}
    try:
        obv = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"])
        obv_series = obv.on_balance_volume()
        recent_vol = float(df["volume"].iloc[-20:].sum())
        if recent_vol > 0:
            out["ex_obv_norm"] = float(obv_series.iloc[-1] / recent_vol)
        if len(df) >= 10 and recent_vol > 0:
            slope = float(obv_series.iloc[-5:].mean() - obv_series.iloc[-10:-5].mean())
            out["ex_obv_slope_norm"] = float(slope / recent_vol)
        score = 50.0
        if out["ex_obv_slope_norm"] > 0: score += 15
        else: score -= 15
        out["ex_obv_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def ad_features(df: pd.DataFrame) -> dict:
    """A/D: Accumulation/Distribution (NORMALIZED)."""
    out = {"ex_ad_norm": 0.0, "ex_ad_slope_norm": 0.0, "ex_ad_score": 50.0}
    try:
        ad = ta.volume.AccDistIndexIndicator(df["high"], df["low"], df["close"], df["volume"])
        ad_series = ad.acc_dist_index()
        recent_vol = float(df["volume"].iloc[-20:].sum())
        if recent_vol > 0:
            out["ex_ad_norm"] = float(ad_series.iloc[-1] / recent_vol)
        if len(df) >= 10 and recent_vol > 0:
            slope = float(ad_series.iloc[-5:].mean() - ad_series.iloc[-10:-5].mean())
            out["ex_ad_slope_norm"] = float(slope / recent_vol)
        score = 50.0
        if out["ex_ad_slope_norm"] > 0: score += 15
        else: score -= 15
        out["ex_ad_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def force_index_features(df: pd.DataFrame) -> dict:
    """Force Index: price change × volume (NORMALIZED)."""
    out = {"ex_force_norm": 0.0, "ex_force_positive": 0, "ex_force_score": 50.0}
    try:
        fi = ta.volume.ForceIndexIndicator(df["close"], df["volume"], window=13)
        v = float(fi.force_index().iloc[-1])
        avg_price = float(df["close"].iloc[-14:].mean())
        avg_vol = float(df["volume"].iloc[-14:].mean())
        norm = avg_price * avg_vol
        if norm > 0:
            out["ex_force_norm"] = float(v / norm)
        out["ex_force_positive"] = int(v > 0)
        score = 50.0
        if v > 0: score += 15
        else: score -= 15
        out["ex_force_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def eom_features(df: pd.DataFrame) -> dict:
    """EOM: Ease of Movement, volume per price change."""
    out = {"ex_eom": 0.0, "ex_eom_positive": 0, "ex_eom_score": 50.0}
    try:
        eom = ta.volume.EaseOfMovementIndicator(
            df["high"], df["low"], df["volume"], window=14
        )
        v = float(eom.ease_of_movement().iloc[-1])
        out["ex_eom"] = v
        out["ex_eom_positive"] = int(v > 0)
        score = 50.0
        if v > 0: score += 10
        else: score -= 10
        out["ex_eom_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def vpt_features(df: pd.DataFrame) -> dict:
    """VPT: Volume Price Trend (NORMALIZED)."""
    out = {"ex_vpt_norm": 0.0, "ex_vpt_slope_norm": 0.0, "ex_vpt_score": 50.0}
    try:
        vpt = ta.volume.VolumePriceTrendIndicator(df["close"], df["volume"])
        vpt_series = vpt.volume_price_trend()
        recent_vol = float(df["volume"].iloc[-20:].sum())
        if recent_vol > 0:
            out["ex_vpt_norm"] = float(vpt_series.iloc[-1] / recent_vol)
        if len(df) >= 10 and recent_vol > 0:
            slope = float(vpt_series.iloc[-5:].mean() - vpt_series.iloc[-10:-5].mean())
            out["ex_vpt_slope_norm"] = float(slope / recent_vol)
        score = 50.0
        if out["ex_vpt_slope_norm"] > 0: score += 15
        else: score -= 15
        out["ex_vpt_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


# ===========================================================================
# VOLATILITY
# ===========================================================================
def keltner_features(df: pd.DataFrame) -> dict:
    """Keltner Channels: ATR-based envelope (alt to BB)."""
    out = {"ex_kc_upper": 0.0, "ex_kc_lower": 0.0, "ex_kc_mid": 0.0,
           "ex_kc_breakout_above": 0, "ex_kc_breakout_below": 0,
           "ex_kc_score": 50.0}
    try:
        kc = ta.volatility.KeltnerChannel(
            df["high"], df["low"], df["close"], window=20, window_atr=10,
            multiplier=2.0
        )
        upper = float(kc.keltner_channel_hband().iloc[-1])
        lower = float(kc.keltner_channel_lband().iloc[-1])
        mid = float(kc.keltner_channel_mband().iloc[-1])
        close = float(df["close"].iloc[-1])
        out["ex_kc_upper"] = upper
        out["ex_kc_lower"] = lower
        out["ex_kc_mid"] = mid
        out["ex_kc_breakout_above"] = int(close > upper)
        out["ex_kc_breakout_below"] = int(close < lower)
        score = 50.0
        if close > upper: score += 15  # breakout
        elif close < lower: score -= 15
        if close > mid: score += 5
        out["ex_kc_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def donchian_features(df: pd.DataFrame) -> dict:
    """Donchian Channels: high/low of period (breakout indicator)."""
    out = {"ex_dc_upper": 0.0, "ex_dc_lower": 0.0, "ex_dc_mid": 0.0,
           "ex_dc_breakout_above": 0, "ex_dc_breakout_below": 0,
           "ex_dc_score": 50.0}
    try:
        dc = ta.volatility.DonchianChannel(
            df["high"], df["low"], df["close"], window=20
        )
        upper = float(dc.donchian_channel_hband().iloc[-1])
        lower = float(dc.donchian_channel_lband().iloc[-1])
        mid = float(dc.donchian_channel_mband().iloc[-1])
        close = float(df["close"].iloc[-1])
        out["ex_dc_upper"] = upper
        out["ex_dc_lower"] = lower
        out["ex_dc_mid"] = mid
        out["ex_dc_breakout_above"] = int(close > upper)
        out["ex_dc_breakout_below"] = int(close < lower)
        score = 50.0
        if close > upper: score += 20
        elif close < lower: score -= 20
        out["ex_dc_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def ulcer_index_features(df: pd.DataFrame) -> dict:
    """Ulcer Index: downside volatility measure."""
    out = {"ex_ulcer": 0.0, "ex_ulcer_score": 50.0}
    try:
        ui = ta.volatility.UlcerIndex(df["close"], window=14)
        v = float(ui.ulcer_index().iloc[-1])
        out["ex_ulcer"] = v
        # Lower is better
        score = 50.0 - max(0, min(40, v * 2))
        out["ex_ulcer_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


def mass_index_features(df: pd.DataFrame) -> dict:
    """Mass Index: reversal detection (range expansion/contraction).
    Manual implementation (ta library returns 0 for some configs)."""
    out = {"ex_mass_index": 0.0, "ex_mass_reversal": 0, "ex_mass_score": 50.0}
    try:
        # Mass Index = sum(EMA(9, high-low)) over 25 periods / EMA(9, EMA(9, high-low))
        hl_range = df["high"] - df["low"]
        ema1 = hl_range.ewm(span=9, adjust=False).mean()
        ema2 = ema1.ewm(span=9, adjust=False).mean()
        ratio = ema1 / ema2.replace(0, 1e-12)
        mass = ratio.rolling(25).sum()
        if not mass.empty and not pd.isna(mass.iloc[-1]):
            v = float(mass.iloc[-1])
            out["ex_mass_index"] = v
            out["ex_mass_reversal"] = int(v > 27)
            score = 50.0
            if v > 27: score += 10
            elif v > 26.5: score += 5
            out["ex_mass_score"] = float(max(0, min(100, score)))
    except Exception:
        pass
    return out


# ===========================================================================
# All-in-one
# ===========================================================================
def compute_all_extended(df: pd.DataFrame) -> Dict:
    """
    Compute ALL extended indicators in one pass.
    Returns a flat dict with all features prefixed `ex_`.
    """
    out = {}
    # Momentum oscillators
    out.update(stochastic_features(df))
    out.update(stochrsi_features(df))
    out.update(williams_r_features(df))
    out.update(cci_features(df))
    out.update(ao_features(df))
    out.update(tsi_features(df))
    out.update(uo_features(df))
    out.update(roc_features(df))
    # Trend strength
    out.update(adx_features(df))
    out.update(aroon_features(df))
    out.update(psar_features(df))
    out.update(vortex_features(df))
    out.update(supertrend_features(df))
    out.update(kst_features(df))
    out.update(trix_features(df))
    # Volume-based
    out.update(mfi_features(df))
    out.update(cmf_features(df))
    out.update(obv_features(df))
    out.update(ad_features(df))
    out.update(force_index_features(df))
    out.update(eom_features(df))
    out.update(vpt_features(df))
    # Volatility
    out.update(keltner_features(df))
    out.update(donchian_features(df))
    out.update(ulcer_index_features(df))
    out.update(mass_index_features(df))
    return out


# Self-test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/toobit-scanner")
    from src.toobit_client import ToobitClient
    import time
    tb = ToobitClient()
    print("Testing on 3 symbols...")
    for sym in ["ALICEUSDT", "TLMUSDT", "HYPERUSDT"]:
        print(f"\n=== {sym} ===")
        try:
            df = tb.get_klines(sym, "4h", 100)
            if df.empty or len(df) < 60:
                print("  too few candles")
                continue
            features = compute_all_extended(df)
            print(f"  Computed {len(features)} features")
            # Group by category
            for k, v in sorted(features.items()):
                if isinstance(v, (int, float)):
                    print(f"  {k:30s}: {v:+.3f}")
        except Exception as e:
            print(f"  error: {e}")
        time.sleep(0.3)
