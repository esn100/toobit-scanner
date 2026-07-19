"""
Generates a self-contained HTML dashboard from data/last_scan.json.
Pumped-up version with new fields (decision, ml_prob, BTC, watchlist).
"""
from __future__ import annotations
import os
import json
import html
from datetime import datetime


def render_dashboard(
    scan_path: str | None = None,
    out_path: str = "dashboard.html",
) -> str:
    if scan_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(here)
        scan_path = os.path.join(project_root, "data", "last_scan.json")
    if not os.path.exists(scan_path):
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("<h1>No scan data yet.</h1>")
        return out_path
    with open(scan_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    weights = data.get("weights", {})
    threshold = data.get("threshold", 75)
    alerts = data.get("alerts", [])
    watchlist = data.get("watchlist", [])
    btc = data.get("btc", {})
    timestamp = data.get("timestamp", "")

    def row(r: dict, decision_class: str) -> str:
        score = r.get("composite_score", 0)
        ml = r.get("ml_prob", 0)
        dec = r.get("decision", "?")
        pat = ", ".join(r.get("patterns", [])) or "—"
        return f"""
        <tr class="{decision_class}">
          <td class="sym">{html.escape(r.get('symbol',''))}</td>
          <td class="score-hi"><b>{score:.1f}</b></td>
          <td>{(ml*100):.0f}%</td>
          <td><span class="pill {dec.lower()}">{dec}</span></td>
          <td>${r.get('market_cap_usd', 0):,.0f}</td>
          <td>${r.get('quote_volume_24h', 0):,.0f}</td>
          <td>{r.get('rsi_value', 0):.1f}</td>
          <td>{r.get('rvol', 0):.2f}x</td>
          <td>{r.get('vwap_distance_pct', 0):+.2f}%</td>
          <td>{r.get('macd_hist', 0):+.4f}</td>
          <td>{r.get('ema_alignment','—')}</td>
          <td>{r.get('momentum_6_pct', 0):+.1f}%</td>
          <td>{html.escape(pat)}</td>
          <td>{r.get('btc_state', btc.get('state', '—'))}</td>
        </tr>
        """

    alert_rows = "".join(row(a, "row-approved") for a in alerts)
    watch_rows = "".join(row(w, "row-watch") for w in watchlist)
    other_rows = "".join(
        row(r, "row-rej")
        for r in data.get("results", [])
        if r.get("decision") not in ("APPROVED", "WATCHLIST")
    )

    avg_score = (
        sum(r.get("composite_score", 0) for r in data.get("results", []))
        / max(1, len(data.get("results", [])))
    )
    top_score = (
        data.get("results", [{}])[0].get("composite_score", 0) if data.get("results") else 0
    )

    weights_str = " ".join(
        f"<span class='pill'>{k.capitalize()}:{v:.0f}</span>"
        for k, v in weights.items()
    )

    btc_state = btc.get("state", "—")
    btc_mod = btc.get("score_modifier", 1.0)
    btc_color = {
        "BULLISH": "var(--hi)", "NEUTRAL": "var(--md)",
        "BEARISH": "var(--warn)", "RISK_OFF": "var(--lo)"
    }.get(btc_state, "var(--fg)")

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PumpHunter-AI — Dashboard</title>
<style>
  :root {{
    --bg: #0b0f17; --fg: #e6edf3; --muted: #8b949e;
    --hi: #2ea043; --md: #d29922; --lo: #6e7681; --warn: #f85149;
    --row: #11161f; --border: #21262d;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: ui-sans-serif, -apple-system, "Segoe UI", sans-serif;
    background: var(--bg); color: var(--fg);
  }}
  header {{
    padding: 24px 32px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 12px;
  }}
  h1 {{ margin: 0; font-size: 20px; }}
  .meta {{ color: var(--muted); font-size: 13px; }}
  .pill {{
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    background: #161b22; border: 1px solid var(--border); margin-left: 6px;
    font-size: 12px;
  }}
  .pill.approved {{ background: #0d2818; color: var(--hi); border-color: #1f3d2a; }}
  .pill.watchlist {{ background: #2a200a; color: var(--md); border-color: #4a3a13; }}
  .pill.rejected {{ background: #2a1414; color: var(--warn); border-color: #4a1f1f; }}
  main {{ padding: 24px 32px; }}
  .cards {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 20px; }}
  .card {{
    background: var(--row); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
  }}
  .card .label {{ color: var(--muted); font-size: 12px; }}
  .card .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  th, td {{
    padding: 7px 8px; border-bottom: 1px solid var(--border);
    text-align: left; vertical-align: top;
  }}
  th {{ color: var(--muted); font-weight: 500; position: sticky; top: 0; background: var(--bg); }}
  tbody tr:hover {{ background: #131a23; }}
  tr.row-approved {{ background: rgba(46, 160, 67, 0.06); }}
  tr.row-watch {{ background: rgba(210, 153, 34, 0.05); }}
  .sym {{ font-weight: 600; }}
  .score-hi {{ color: var(--hi); }}
  .muted {{ color: var(--muted); font-size: 11px; }}
  h2.section {{ font-size: 16px; margin: 24px 0 8px 0; }}
  .weights {{ display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>🤖 PumpHunter-AI</h1>
    <div class="meta">Last run: {timestamp}</div>
  </div>
  <div>
    <span class="pill">BTC: <b style="color: {btc_color}">{btc_state}</b> (mod {btc_mod:.2f})</span>
    <span class="pill">Threshold: {threshold}</span>
  </div>
</header>
<main>
  <div class="cards">
    <div class="card"><div class="label">Scanned</div><div class="value">{data.get('scanned', 0)}</div></div>
    <div class="card"><div class="label">APPROVED</div><div class="value" style="color: var(--hi)">{len(alerts)}</div></div>
    <div class="card"><div class="label">WATCHLIST</div><div class="value" style="color: var(--md)">{len(watchlist)}</div></div>
    <div class="card"><div class="label">Avg score</div><div class="value">{avg_score:.1f}</div></div>
    <div class="card"><div class="label">Top score</div><div class="value">{top_score:.1f}</div></div>
  </div>

  <div class="meta" style="margin-bottom: 8px;">Rule weights:</div>
  <div class="weights">{weights_str}</div>

  <h2 class="section">🚨 Approved</h2>
  <table>
    <thead>
      <tr>
        <th>Symbol</th><th>Score</th><th>ML%</th><th>Decision</th>
        <th>MCap</th><th>24h vol</th><th>RSI</th><th>RVol</th>
        <th>VWAP Δ%</th><th>MACD</th><th>EMA</th><th>M6%</th>
        <th>Patterns</th><th>BTC</th>
      </tr>
    </thead>
    <tbody>
      {alert_rows if alert_rows else '<tr><td colspan="14">No approved signals in this run.</td></tr>'}
    </tbody>
  </table>

  <h2 class="section">👀 Watchlist</h2>
  <table>
    <thead>
      <tr>
        <th>Symbol</th><th>Score</th><th>ML%</th><th>Decision</th>
        <th>MCap</th><th>24h vol</th><th>RSI</th><th>RVol</th>
        <th>VWAP Δ%</th><th>MACD</th><th>EMA</th><th>M6%</th>
        <th>Patterns</th><th>BTC</th>
      </tr>
    </thead>
    <tbody>
      {watch_rows if watch_rows else '<tr><td colspan="14">No watchlist items.</td></tr>'}
    </tbody>
  </table>

  <h2 class="section">📊 All scanned</h2>
  <table>
    <thead>
      <tr>
        <th>Symbol</th><th>Score</th><th>ML%</th><th>Decision</th>
        <th>MCap</th><th>24h vol</th><th>RSI</th><th>RVol</th>
        <th>VWAP Δ%</th><th>MACD</th><th>EMA</th><th>M6%</th>
        <th>Patterns</th><th>BTC</th>
      </tr>
    </thead>
    <tbody>
      {other_rows if other_rows else '<tr><td colspan="14">No other symbols.</td></tr>'}
    </tbody>
  </table>
</main>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return out_path


if __name__ == "__main__":
    p = render_dashboard()
    print(f"Dashboard written to {p}")
