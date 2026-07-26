"""
Advanced ML models for PumpHunter-AI.

Trains:
  1. XGBoost classifier
  2. LightGBM classifier
  3. Random Forest with hyperparameter tuning

Compares with logistic regression baseline.
Selects best model based on CV F1.
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import precision_recall_curve, f1_score


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def load_data() -> tuple:
    path = os.path.join(DATA_DIR, "extended_timeseries.csv")
    if not os.path.exists(path):
        return None, None, None
    df = pd.read_csv(path)
    df = df.dropna(subset=["fwd_12h"])
    df["pump"] = (df["fwd_12h"] > 3).astype(int)
    feat_cols = [c for c in df.columns if c.startswith(("ind_", "candle_", "ex_"))
                 and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feat_cols].fillna(0).values
    y = df["pump"].values
    valid = X.std(axis=0) > 1e-9
    X = X[:, valid]
    valid_feats = [f for f, v in zip(feat_cols, valid) if v]
    return X, y, valid_feats


def train_all_models(X: np.ndarray, y: np.ndarray) -> Dict:
    """Train multiple models and compare."""
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Logistic Regression (baseline)
    print("\n--- 1. Logistic Regression (baseline) ---")
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                           C=0.5, random_state=42)
    lr.fit(Xs, y)
    f1 = cross_val_score(lr, Xs, y, cv=cv, scoring="f1")
    acc = cross_val_score(lr, Xs, y, cv=cv, scoring="accuracy")
    auc = cross_val_score(lr, Xs, y, cv=cv, scoring="roc_auc")
    results["logreg"] = {
        "f1": float(f1.mean()),
        "acc": float(acc.mean()),
        "auc": float(auc.mean()),
        "model": lr,
        "scaler": sc,
    }
    print(f"   F1={f1.mean():.3f} Acc={acc.mean():.3f} AUC={auc.mean():.3f}")

    # 2. Random Forest
    print("\n--- 2. Random Forest ---")
    rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                 class_weight="balanced", random_state=42,
                                 n_jobs=-1)
    rf.fit(X, y)
    f1 = cross_val_score(rf, X, y, cv=cv, scoring="f1")
    acc = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
    auc = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    results["random_forest"] = {
        "f1": float(f1.mean()),
        "acc": float(acc.mean()),
        "auc": float(auc.mean()),
        "model": rf,
    }
    print(f"   F1={f1.mean():.3f} Acc={acc.mean():.3f} AUC={auc.mean():.3f}")

    # 3. Gradient Boosting
    print("\n--- 3. Gradient Boosting (sklearn) ---")
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                    learning_rate=0.1, random_state=42)
    gb.fit(X, y)
    f1 = cross_val_score(gb, X, y, cv=cv, scoring="f1")
    acc = cross_val_score(gb, X, y, cv=cv, scoring="accuracy")
    auc = cross_val_score(gb, X, y, cv=cv, scoring="roc_auc")
    results["gradient_boosting"] = {
        "f1": float(f1.mean()),
        "acc": float(acc.mean()),
        "auc": float(auc.mean()),
        "model": gb,
    }
    print(f"   F1={f1.mean():.3f} Acc={acc.mean():.3f} AUC={auc.mean():.3f}")

    # 4. XGBoost
    if HAS_XGB:
        print("\n--- 4. XGBoost ---")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=sum(y==0)/max(sum(y==1), 1),
            random_state=42, n_jobs=-1, eval_metric="logloss",
        )
        xgb_model.fit(X, y)
        f1 = cross_val_score(xgb_model, X, y, cv=cv, scoring="f1")
        acc = cross_val_score(xgb_model, X, y, cv=cv, scoring="accuracy")
        auc = cross_val_score(xgb_model, X, y, cv=cv, scoring="roc_auc")
        results["xgboost"] = {
            "f1": float(f1.mean()),
            "acc": float(acc.mean()),
            "auc": float(auc.mean()),
            "model": xgb_model,
        }
        print(f"   F1={f1.mean():.3f} Acc={acc.mean():.3f} AUC={auc.mean():.3f}")
    else:
        print("   XGBoost not installed, skipping")

    # 5. LightGBM
    if HAS_LGB:
        print("\n--- 5. LightGBM ---")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            class_weight="balanced", random_state=42, n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(X, y)
        f1 = cross_val_score(lgb_model, X, y, cv=cv, scoring="f1")
        acc = cross_val_score(lgb_model, X, y, cv=cv, scoring="accuracy")
        auc = cross_val_score(lgb_model, X, y, cv=cv, scoring="roc_auc")
        results["lightgbm"] = {
            "f1": float(f1.mean()),
            "acc": float(acc.mean()),
            "auc": float(auc.mean()),
            "model": lgb_model,
        }
        print(f"   F1={f1.mean():.3f} Acc={acc.mean():.3f} AUC={auc.mean():.3f}")
    else:
        print("   LightGBM not installed, skipping")

    return results


def find_best_model(results: Dict) -> tuple:
    """Find best model by AUC."""
    best_name = max(results, key=lambda k: results[k]["auc"])
    return best_name, results[best_name]


def per_symbol_advanced(df: pd.DataFrame, feat_cols: list) -> Dict:
    """Train per-symbol advanced models."""
    per_sym_results = {}
    for sym, grp in df.groupby("symbol"):
        if len(grp) < 80 or grp["pump"].sum() < 15:
            continue
        X = grp[feat_cols].fillna(0).values
        y = grp["pump"].values
        valid = X.std(axis=0) > 1e-9
        X = X[:, valid]
        if X.shape[1] == 0:
            continue
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        # Try multiple models
        scores = {}
        # RF
        rf = RandomForestClassifier(n_estimators=100, max_depth=6,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
        f1 = cross_val_score(rf, X, y, cv=cv, scoring="f1").mean()
        scores["rf"] = float(f1)
        # GB
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                        learning_rate=0.1, random_state=42)
        f1 = cross_val_score(gb, X, y, cv=cv, scoring="f1").mean()
        scores["gb"] = float(f1)
        # XGB
        if HAS_XGB:
            xgb_m = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                scale_pos_weight=sum(y==0)/max(sum(y==1), 1),
                random_state=42, n_jobs=-1, eval_metric="logloss",
            )
            f1 = cross_val_score(xgb_m, X, y, cv=cv, scoring="f1").mean()
            scores["xgb"] = float(f1)
        best = max(scores, key=scores.get)
        # Train best on full data
        if best == "rf":
            model = RandomForestClassifier(n_estimators=100, max_depth=6,
                                          class_weight="balanced", random_state=42)
        elif best == "gb":
            model = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                              learning_rate=0.1, random_state=42)
        elif best == "xgb" and HAS_XGB:
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                scale_pos_weight=sum(y==0)/max(sum(y==1), 1),
                random_state=42, eval_metric="logloss",
            )
        model.fit(X, y)
        # Feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = np.zeros(X.shape[1])
        valid_feats = [f for f, v in zip(feat_cols, valid) if v]
        top_idx = np.argsort(importances)[::-1][:10]
        top_features = [(valid_feats[i], float(importances[i])) for i in top_idx if importances[i] > 0]
        per_sym_results[sym] = {
            "n": int(len(grp)),
            "pump_rate": float(y.mean()),
            "best_model": best,
            "scores": scores,
            "best_score": float(max(scores.values())),
            "top_features": top_features,
        }
    return per_sym_results


def main():
    X, y, feat_cols = load_data()
    if X is None:
        print("No data found")
        return
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, {y.mean()*100:.1f}% pump rate")

    print("\n" + "="*80)
    print("GLOBAL MODELS COMPARISON")
    print("="*80)
    results = train_all_models(X, y)
    best_name, best = find_best_model(results)
    print(f"\n>>> Best global model: {best_name} (AUC={best['auc']:.3f})")
    # Save
    out = {
        "global_models": {k: {"f1": v["f1"], "acc": v["acc"], "auc": v["auc"]}
                          for k, v in results.items()},
        "best_global": best_name,
    }
    # Per-symbol
    print("\n" + "="*80)
    print("PER-SYMBOL ADVANCED MODELS")
    print("="*80)
    df = pd.read_csv(os.path.join(DATA_DIR, "extended_timeseries.csv"))
    df = df.dropna(subset=["fwd_12h"])
    df["pump"] = (df["fwd_12h"] > 3).astype(int)
    per_sym = per_symbol_advanced(df, feat_cols)
    out["per_symbol"] = per_sym
    for sym, m in sorted(per_sym.items(), key=lambda x: -x[1]["best_score"]):
        pos = ", ".join([f"{p[0].split('_')[-1]}({p[1]:.2f})" for p in m['top_features'][:3]])
        print(f"  {sym:<12} best={m['best_model']:<5} F1={m['best_score']:.3f} pump={m['pump_rate']*100:.0f}%  {pos}")
    with open(os.path.join(DATA_DIR, "advanced_ml_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to data/advanced_ml_results.json")


if __name__ == "__main__":
    main()
