"""
Self-training loop for PumpHunter-AI.

Runs every hour, automatically:
  1. Collects outcomes from DB
  2. If < 30 signals, lowers threshold to gather more
  3. If >= 30 signals, calibrates weights via:
     - Logistic regression on all features
     - Top correlated features
     - Best TP/SL via grid search
  4. Saves best params + updates config
  5. Logs progress

Usage:
    python -m src.self_train --once   # run one cycle
    python -m src.self_train          # loop every hour
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STATE_FILE = os.path.join(DATA_DIR, "self_train_state.json")
LOG_FILE = os.path.join(DATA_DIR, "self_train.log")

MIN_SIGNALS_FOR_CALIBRATION = 30
MIN_SIGNALS_FOR_TUNING = 10


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "last_run": None,
        "iterations": 0,
        "total_signals_seen": 0,
        "calibration_history": [],
        "current_params": {
            "tp_pct": 5.0,
            "sl_pct": 3.0,
            "trail_pct": 1.5,
            "conf_threshold": 60,
        },
    }


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def get_outcomes_and_features() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load all features and labeled outcomes from DB."""
    # Features from CSV
    feat_path = os.path.join(DATA_DIR, "feature_log.csv")
    if not os.path.exists(feat_path):
        return pd.DataFrame(), pd.DataFrame()
    df_feat = pd.read_csv(feat_path)
    # Outcomes from CSV
    out_path = os.path.join(DATA_DIR, "outcome_log.csv")
    if not os.path.exists(out_path):
        return df_feat, pd.DataFrame()
    df_out = pd.read_csv(out_path)
    return df_feat, df_out


def build_training_set() -> pd.DataFrame:
    """Merge features with their 12h outcomes, label=1 if pump, 0 otherwise."""
    df_feat, df_out = get_outcomes_and_features()
    if df_feat.empty or df_out.empty:
        return pd.DataFrame()
    df_feat["ts"] = pd.to_datetime(df_feat["ts"], utc=True, errors="coerce")
    df_out["ts"] = pd.to_datetime(df_out["ts"], utc=True, errors="coerce")
    # Outcomes: ts = feature time, signal_close = price then, horizon_close = 12h later
    df_out["label_pump"] = (df_out["return_pct"] >= 3.0).astype(int)
    # Direct merge on (ts, symbol)
    merged = df_feat.merge(
        df_out[["ts", "symbol", "label_pump", "return_pct"]],
        on=["ts", "symbol"], how="inner"
    )
    if merged.empty:
        return pd.DataFrame()
    # Drop rows with NaN features
    merged = merged.dropna(subset=["label_pump"])
    return merged


def calibrate_thresholds(df: pd.DataFrame) -> dict:
    """
    Find best confidence threshold via logistic regression.
    Returns best threshold + weight adjustments.
    """
    if df.empty or len(df) < MIN_SIGNALS_FOR_TUNING:
        return {"status": "insufficient_data", "n": len(df)}
    # Get feature columns (numeric, ex_/f_/ind_/a_/m_/s_)
    feat_cols = []
    for c in df.columns:
        if c.startswith(("ex_", "f_", "ind_", "a_", "m_", "s_")) and c not in ("ex_",):
            if pd.api.types.is_numeric_dtype(df[c]):
                feat_cols.append(c)
    if not feat_cols:
        return {"status": "no_features"}
    X = df[feat_cols].fillna(0.0).values
    y = df["label_pump"].values
    # Drop rows with all-zero features
    valid = ~np.all(X == 0, axis=1)
    X = X[valid]
    y = y[valid]
    if len(y) < 10 or y.sum() < 3:
        return {"status": "too_few_pumps", "n": len(y), "pumps": int(y.sum())}
    # Standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # Logistic regression
    try:
        logreg = LogisticRegression(max_iter=2000, class_weight="balanced",
                                    C=0.5, random_state=42)
        logreg.fit(Xs, y)
        # Cross-validate
        cv_f1 = cross_val_score(logreg, Xs, y, cv=3, scoring="f1")
        cv_acc = cross_val_score(logreg, Xs, y, cv=3, scoring="accuracy")
        # Top features by coefficient
        coef_pairs = sorted(
            zip(feat_cols, logreg.coef_[0]),
            key=lambda x: abs(x[1]), reverse=True
        )
        top_positive = [(f, c) for f, c in coef_pairs if c > 0][:5]
        top_negative = [(f, c) for f, c in coef_pairs if c < 0][:5]
        # === NEW: Find optimal probability threshold ===
        from sklearn.metrics import precision_recall_curve, f1_score
        y_proba = logreg.predict_proba(Xs)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_proba)
        # Find threshold that maximizes F1
        f1_scores = 2 * (precisions * recalls) / np.clip(precisions + recalls, 1e-9, None)
        best_idx = int(np.argmax(f1_scores[:-1]))  # exclude last (no threshold)
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])
        # === NEW: Identify weak features (coeff close to 0) ===
        weak_features = []
        for f, c in coef_pairs:
            if abs(c) < 0.1:  # essentially zero
                weak_features.append(f)
        return {
            "status": "ok",
            "n_samples": int(len(y)),
            "n_pumps": int(y.sum()),
            "pump_rate": float(y.mean()),
            "cv_f1_mean": float(cv_f1.mean()),
            "cv_acc_mean": float(cv_acc.mean()),
            "top_positive": [{"feature": f, "coef": float(c)} for f, c in top_positive],
            "top_negative": [{"feature": f, "coef": float(c)} for f, c in top_negative],
            "best_proba_threshold": best_threshold,
            "best_proba_f1": best_f1,
            "weak_features": weak_features[:20],  # top 20 to drop
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def grid_search_tp_sl(df: pd.DataFrame) -> dict:
    """Find best TP/SL/trail using historical outcomes with smart v2 simulation."""
    if df.empty or len(df) < MIN_SIGNALS_FOR_TUNING:
        return {"status": "insufficient_data"}
    if "return_pct" not in df.columns:
        return {"status": "no_return"}
    # Simulate trades with smart v2 logic
    results = []
    for tp in [2.0, 2.5, 3.0, 5.0, 7.0]:
        for sl in [1.5, 2.0, 3.0]:
            for trail in [0.5, 1.0, 1.5]:
                # Simulate with breakeven lock at +1%, profit lock at +2%
                wins = 0
                losses = 0
                pnl_total = 0
                for _, row in df.iterrows():
                    ret = row["return_pct"]
                    # Smart v2 simplified:
                    # If returns > tp, TP hit
                    # If returns > +1%, breakeven (lock at entry = 0 P&L)
                    # If returns > +2%, lock at +1% P&L
                    # If returns < -sl, full SL hit
                    if ret >= tp:
                        wins += 1
                        pnl_total += tp
                    elif ret >= 2.0:  # +2% reached
                        wins += 1
                        pnl_total += 1.0  # locked at +1%
                    elif ret >= 1.0:  # +1% reached
                        wins += 1
                        pnl_total += 0  # breakeven
                    elif ret <= -sl:
                        losses += 1
                        pnl_total -= sl
                    else:
                        # Mid range: close at return
                        pnl_total += ret
                n = len(df)
                wr = wins / n if n > 0 else 0
                results.append({
                    "tp": tp, "sl": sl, "trail": trail,
                    "n": n, "wins": int(wins), "losses": int(losses),
                    "wr": wr, "pnl": pnl_total,
                })
    results.sort(key=lambda x: (x["wr"], x["pnl"]), reverse=True)
    return {"status": "ok", "best": results[:5], "all": results}


def feature_selection(df: pd.DataFrame, calibration: dict) -> list:
    """Identify features to keep based on calibration results."""
    if calibration.get("status") != "ok":
        return []
    weak = set(calibration.get("weak_features", []))
    # Also: drop features with no variance
    feat_cols = []
    for c in df.columns:
        if c.startswith(("ex_", "f_", "ind_", "a_", "m_", "s_")):
            if pd.api.types.is_numeric_dtype(df[c]):
                if c in weak:
                    continue
                if df[c].std() < 1e-6:
                    continue
                feat_cols.append(c)
    return feat_cols


def adaptive_strategy_adjustment(state: dict, calibration: dict,
                                  tp_sl: dict) -> dict:
    """
    Adjust strategy based on calibration results.

    Goal hierarchy (priority order):
      1. Win rate >= 70% (quality > quantity)
      2. P&L positive
      3. Reasonable signal count
    """
    if calibration.get("status") != "ok":
        return state["current_params"]
    current = state["current_params"]
    # Strategy 1: Adjust confidence threshold
    pump_rate = calibration.get("pump_rate", 0.16)
    if pump_rate < 0.10:
        # Too few pumps, lower threshold to find more
        new_conf = max(40, current.get("conf_threshold", 60) - 5)
    elif pump_rate > 0.30:
        # Many pumps, be more selective
        new_conf = min(80, current.get("conf_threshold", 60) + 3)
    else:
        new_conf = current.get("conf_threshold", 60)
    # Strategy 2: Find TP/SL with BEST win rate (not just PnL)
    if tp_sl and tp_sl.get("status") == "ok" and tp_sl.get("all"):
        # Re-sort by win rate first, then PnL
        sorted_by_wr = sorted(tp_sl["all"],
                              key=lambda x: (x["wr"], x["pnl"]),
                              reverse=True)
        # Take top 3, then choose the one with WR >= 50%
        best = None
        for cand in sorted_by_wr[:5]:
            if cand["wr"] >= 0.5:
                best = cand
                break
        if best is None:
            # If no 50% WR, take best by WR
            best = sorted_by_wr[0]
        return {
            "tp_pct": best["tp"],
            "sl_pct": best["sl"],
            "trail_pct": best["trail"],
            "conf_threshold": new_conf,
        }
    return {
        "tp_pct": current.get("tp_pct", 5.0),
        "sl_pct": current.get("sl_pct", 3.0),
        "trail_pct": current.get("trail_pct", 1.5),
        "conf_threshold": new_conf,
    }


def adaptive_threshold_mode(state: dict, n_signals: int) -> dict:
    """
    If we have too few signals, lower threshold to get more data.
    This is the COLLECT phase.
    """
    if n_signals >= MIN_SIGNALS_FOR_CALIBRATION:
        return state["current_params"]
    # We need more data. Lower threshold.
    current = state["current_params"]
    # Reduce conf threshold by 5
    new_conf = max(40, current.get("conf_threshold", 60) - 5)
    return {
        "tp_pct": current.get("tp_pct", 5.0),
        "sl_pct": current.get("sl_pct", 3.0),
        "trail_pct": current.get("trail_pct", 1.5),
        "conf_threshold": new_conf,
        "collecting": True,
    }


def save_params_to_config(params: dict):
    """Write params to a config file that auto-trader can read."""
    cfg_path = os.path.join(DATA_DIR, "self_trained_params.json")
    with open(cfg_path, "w") as f:
        json.dump(params, f, indent=2)


def run_iteration(verbose: bool = True) -> dict:
    """One full training iteration."""
    state = load_state()
    state["iterations"] += 1
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    if verbose:
        log(f"=== Iteration #{state['iterations']} ===")
    # 1) Build training set
    df = build_training_set()
    if verbose:
        log(f"Training samples: {len(df)}, pumps: "
            f"{int(df['label_pump'].sum()) if not df.empty else 0}")
    state["total_signals_seen"] = len(df)
    # 2) If we have enough data, calibrate
    calibration = None
    tp_sl_results = None
    feature_list = None
    if len(df) >= MIN_SIGNALS_FOR_CALIBRATION:
        if verbose:
            log(f"  -> Enough data ({len(df)} >= "
                f"{MIN_SIGNALS_FOR_CALIBRATION}), calibrating...")
        calibration = calibrate_thresholds(df)
        tp_sl_results = grid_search_tp_sl(df)
        feature_list = feature_selection(df, calibration)
        if verbose:
            if calibration.get("status") == "ok":
                log(f"  -> LR CV F1: {calibration['cv_f1_mean']:.3f}, "
                    f"Acc: {calibration['cv_acc_mean']:.3f}")
                log(f"  -> Pump rate: {calibration['pump_rate']*100:.1f}%, "
                    f"Best proba threshold: "
                    f"{calibration.get('best_proba_threshold', 0.5):.3f}")
                if calibration.get("top_positive"):
                    pos = ", ".join([f"{p['feature']}({p['coef']:+.2f})"
                                     for p in calibration["top_positive"][:3]])
                    log(f"  -> Top positive: {pos}")
                if calibration.get("top_negative"):
                    neg = ", ".join([f"{p['feature']}({p['coef']:+.2f})"
                                     for p in calibration["top_negative"][:3]])
                    log(f"  -> Top negative: {neg}")
                if calibration.get("weak_features"):
                    log(f"  -> Weak features to drop: "
                        f"{len(calibration['weak_features'])}")
            if tp_sl_results and tp_sl_results.get("status") == "ok":
                best = tp_sl_results["best"][0]
                log(f"  -> Best TP/SL: TP={best['tp']}%, SL={best['sl']}%, "
                    f"trail={best['trail']}%, WR={best['wr']*100:.1f}%, "
                    f"PnL={best['pnl']:+.1f}%")
        # === ADAPTIVE ADJUSTMENT ===
        new_params = adaptive_strategy_adjustment(state, calibration,
                                                   tp_sl_results)
        state["current_params"] = new_params
        save_params_to_config(new_params)
        if verbose:
            log(f"  -> New params: {new_params}")
            log(f"  -> Active features: {len(feature_list) if feature_list else 0}")
    else:
        # Collecting phase - lower threshold
        new_params = adaptive_threshold_mode(state, len(df))
        if new_params.get("collecting"):
            if verbose:
                log(f"  -> Need more data. Lowered conf threshold to "
                    f"{new_params['conf_threshold']}")
        state["current_params"] = new_params
        save_params_to_config(new_params)
    # Save calibration result
    state["calibration_history"].append({
        "ts": state["last_run"],
        "n_samples": len(df),
        "n_pumps": int(df["label_pump"].sum()) if not df.empty else 0,
        "calibration": calibration,
        "tp_sl": tp_sl_results,
        "feature_list": feature_list,
        "params": state["current_params"],
    })
    # Trim history
    state["calibration_history"] = state["calibration_history"][-100:]
    save_state(state)
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true",
                   help="Run one iteration and exit")
    p.add_argument("--interval", type=int, default=3600,
                   help="Seconds between iterations (default 3600 = 1 hour)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    if args.once:
        run_iteration(verbose=not args.quiet)
        return
    log(f"Starting self-training loop (interval={args.interval}s)")
    while True:
        try:
            run_iteration(verbose=not args.quiet)
        except Exception as e:
            log(f"ERROR: {e}")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            log("Stopping")
            return


if __name__ == "__main__":
    main()
