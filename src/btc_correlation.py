"""
BTC correlation and intermarket features.

For small caps, finding symbols that move INDEPENDENTLY of BTC
(within a 4h window) is one of the strongest alpha signals - it
means the move is driven by symbol-specific news/demand, not
broad market beta.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict


def btc_correlation_features(
    sym_df: pd.DataFrame, btc_df: pd.DataFrame,
    windows: tuple = (12, 24, 48),  # 12*4h=2d, 24=4d, 48=8d
) -> Dict:
    """
    Compute rolling correlation with BTC.
    Returns: correlation at each window, beta, and an
    'independent_mover' flag.
    """
    if sym_df.empty or btc_df.empty or len(sym_df) < 50 or len(btc_df) < 50:
        return {
            "btc_corr_2d": 0.0, "btc_corr_4d": 0.0, "btc_corr_8d": 0.0,
            "btc_beta": 1.0, "btc_corr_change": 0.0,
            "independent_mover": False,
        }
    # Align timestamps
    sym = sym_df.set_index("open_time")["close"].astype(float)
    btc = btc_df.set_index("open_time")["close"].astype(float)
    # Find common timestamps
    common = sym.index.intersection(btc.index)
    if len(common) < 50:
        return {
            "btc_corr_2d": 0.0, "btc_corr_4d": 0.0, "btc_corr_8d": 0.0,
            "btc_beta": 1.0, "btc_corr_change": 0.0,
            "independent_mover": False,
        }
    sym = sym.loc[common]
    btc = btc.loc[common]
    sym_ret = sym.pct_change().fillna(0)
    btc_ret = btc.pct_change().fillna(0)
    out = {}
    for w in windows:
        if len(sym_ret) < w:
            out[f"btc_corr_{w // 6}d"] = 0.0
            continue
        s = sym_ret.tail(w).values
        b = btc_ret.tail(w).values
        try:
            corr = float(np.corrcoef(s, b)[0, 1])
        except Exception:
            corr = 0.0
        out[f"btc_corr_{w // 6}d"] = corr
    # Beta: covariance / variance
    try:
        cov = float(np.cov(sym_ret.tail(48), btc_ret.tail(48))[0, 1])
        var = float(np.var(btc_ret.tail(48)))
        beta = cov / var if var > 0 else 1.0
    except Exception:
        beta = 1.0
    out["btc_beta"] = beta
    # Correlation change: recent vs longer window
    c_short = out.get("btc_corr_2d", 0)
    c_long = out.get("btc_corr_8d", 0)
    out["btc_corr_change"] = c_short - c_long
    # Independent mover: low correlation AND positive recent return
    sym_return_4h = float((sym.iloc[-1] - sym.iloc[-2]) / max(sym.iloc[-2], 1e-9) * 100)
    out["independent_mover"] = bool(
        c_short < 0.4 and sym_return_4h > 1.0
    )
    out["sym_return_4h_pct"] = sym_return_4h
    return out
