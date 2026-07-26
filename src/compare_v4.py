"""Compare all backtest v4 runs and recommend best params."""
import json
import os
import glob
import pandas as pd

# Find all v4_results.json
results = []
for path in glob.glob("backtest_v4*/v4_results.json"):
    with open(path) as f:
        try:
            r = json.load(f)
            r["_path"] = path
            results.append(r)
        except Exception:
            continue

if not results:
    print("No v4_results.json found")
    raise SystemExit(1)

df = pd.DataFrame([{
    "name": r["_path"].split("/")[0],
    "n_trades": r["n_trades"],
    "win_rate": r["win_rate"] * 100,
    "total_pnl": r["total_pnl_pct"],
    "avg_pnl": r["avg_pnl_pct"],
    "profit_factor": r["profit_factor"],
    "tp": r["params"]["tp_pct"],
    "sl": r["params"]["sl_pct"],
    "trail": r["params"]["trailing_pct"],
    "days": r["params"]["days"],
} for r in results])
df = df.sort_values("win_rate", ascending=False)
print("=" * 100)
print(f"{'Name':<30} {'N':>4} {'WR%':>6} {'Total':>8} {'Avg':>7} {'PF':>7} {'TP':>5} {'SL':>5} {'Tr':>4} {'Days':>5}")
print("=" * 100)
for _, r in df.iterrows():
    print(f"{r['name']:<30} {r['n_trades']:>4.0f} {r['win_rate']:>6.1f} {r['total_pnl']:>+7.1f}% {r['avg_pnl']:>+6.2f}% {r['profit_factor']:>7.2f} {r['tp']:>5.1f} {r['sl']:>5.1f} {r['trail']:>4.1f} {r['days']:>5.0f}")
print("=" * 100)

# Best by win rate
print()
print("BEST BY WIN RATE:")
best_wr = df.iloc[0]
print(f"  {best_wr['name']}: {best_wr['win_rate']:.1f}% win rate, "
      f"TP={best_wr['tp']}%, SL={best_wr['sl']}%, trail={best_wr['trail']}%, "
      f"days={best_wr['days']:.0f}, n={best_wr['n_trades']:.0f}")

# Best by profit factor
best_pf = df.sort_values("profit_factor", ascending=False).iloc[0]
print(f"BEST BY PROFIT FACTOR:")
print(f"  {best_pf['name']}: PF={best_pf['profit_factor']:.1f}, "
      f"TP={best_pf['tp']}%, SL={best_pf['sl']}%, trail={best_pf['trail']}%, "
      f"win_rate={best_pf['win_rate']:.1f}%")
