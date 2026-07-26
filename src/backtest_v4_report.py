"""Generate human-readable report from v4_results.json"""
import json
import os
import sys

base = sys.argv[1] if len(sys.argv) > 1 else "backtest_v4_90d"

with open(f"{base}/v4_results.json") as f:
    s = json.load(f)

lines = [
    f"# PumpHunter Backtest v4 — {base}",
    "",
    f"**Generated:** {s['timestamp']}",
    f"**BTC state:** {s['btc_state']}",
    "",
    "## Overall",
    f"- **Snapshots:** {s['n_snapshots']}",
    f"- **Ultra signals:** {s['n_trades']} ({s['n_trades']/s['n_snapshots']*100:.1f}%)",
    f"- **Wins:** {s['n_wins']}",
    f"- **Losses:** {s['n_losses']}",
    f"- **Breakeven:** {s['n_breakeven']}",
    "",
    "## Performance",
    f"- **Win rate:** {s['win_rate']*100:.1f}%",
    f"- **Total P&L:** {s['total_pnl_pct']:+.2f}%",
    f"- **Avg P&L:** {s['avg_pnl_pct']:+.2f}%",
    f"- **Profit factor:** {s['profit_factor']:.2f}",
    "",
    "## Filter Pass Rates",
    f"- Prefilter: {s['filter_pass']['prefilter']} ({s['filter_pass']['prefilter']/s['filter_pass']['from_total']*100:.1f}%)",
    f"- +Anti-late: {s['filter_pass']['prefilter+anti_late']} ({s['filter_pass']['prefilter+anti_late']/s['filter_pass']['from_total']*100:.1f}%)",
    f"- +Ultra-strict: {s['filter_pass']['ultra']} ({s['filter_pass']['ultra']/s['filter_pass']['from_total']*100:.1f}%)",
    "",
    "## By Direction",
    "| Direction | N | Win Rate | Avg P&L | Total P&L |",
    "|---|---:|---:|---:|---:|",
]
for d, ds in s.get('by_direction', {}).items():
    lines.append(f"| {d} | {ds['n']} | {ds['win_rate']*100:.1f}% | {ds['avg_pnl']:+.2f}% | {ds['total_pnl']:+.2f}% |")
lines += [
    "",
    "## By Exit Reason",
    "| Reason | N | Wins | Avg P&L |",
    "|---|---:|---:|---:|",
]
for r, rs in s.get('by_exit_reason', {}).items():
    lines.append(f"| {r} | {rs['n']} | {rs['wins']} | {rs['avg_pnl']:+.2f}% |")
lines += [
    "",
    "## Parameters",
    f"- Days: {s['params']['days']}",
    f"- TP: {s['params']['tp_pct']}%",
    f"- SL: {s['params']['sl_pct']}%",
    f"- Trailing: {s['params']['trailing_pct']}%",
    f"- Max hold: {s['params']['max_hold_bars']*4}h",
    f"- Snapshot every: {s['params']['snapshot_every_bars']*4}h",
]

out = f"{base}/v4_report.md"
with open(out, "w") as f:
    f.write("\n".join(lines))
print(f"Wrote {out}")
