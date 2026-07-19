"""
Generates a self-contained HTML dashboard from data/last_scan.json.
Just open dashboard.html in a browser.
"""
from __future__ import annotations
import os
import json
import html
from datetime import datetime


def render_dashboard(scan_path: str = "data/last_scan.json", out_path: str = "dashboard.html") -> str:
    if not os.path.exists(scan_path):
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("<h1>No scan data yet.</h1>")
        return out_path
    with open(scan_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for r in data.get("results", []):
        score = r["score"]
        cls = (
            "score-hi" if score >= 90 else
            "score-md" if score >= 75 else
            "score-lo"
        )
        patterns = ", ".join(r.get("patterns", [])) or "—"
        rows.append(f"""
        <tr>
          <td class="sym">{html.escape(r['symbol'])}</td>
          <td class="{cls}"><b>{score:.1f}</b></td>
          <td>${r.get('market_cap_usd', 0):,.0f}</td>
          <td>${r.get('quote_volume_24h', 0):,.0f}</td>
          <td>{r.get('rsi_value', 0):.1f}<br><span class="muted">{r.get('rsi_divergence','')}</span></td>
          <td>{r.get('macd_hist', 0):+.4f}<br><span class="muted">{r.get('macd_divergence','')}</span></td>
          <td>{r.get('ema_alignment','—')}</td>
          <td>{html.escape(patterns)}</td>
          <td>{r.get('social_score', 0):.0f}</td>
          <td>{r.get('liq_bias', 0):+.2f}</td>
          <td>{r.get('price_change_pct_24h', 0):+.1f}%</td>
        </tr>
        """)

    weights = data.get("weights", {})
    threshold = data.get("threshold", 85)
    alerts = data.get("alerts", [])
    timestamp = data.get("timestamp", "")

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Toobit Scanner — Dashboard</title>
<style>
  :root {{
    --bg: #0b0f17; --fg: #e6edf3; --muted: #8b949e;
    --hi: #2ea043; --md: #d29922; --lo: #6e7681;
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
  }}
  h1 {{ margin: 0; font-size: 20px; }}
  .meta {{ color: var(--muted); font-size: 13px; }}
  .pill {{
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    background: #161b22; border: 1px solid var(--border); margin-left: 6px;
  }}
  main {{ padding: 24px 32px; }}
  .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }}
  .card {{
    background: var(--row); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
  }}
  .card .label {{ color: var(--muted); font-size: 12px; }}
  .card .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  table {{
    width: 100%; border-collapse: collapse; font-size: 13px;
  }}
  th, td {{
    padding: 8px 10px; border-bottom: 1px solid var(--border);
    text-align: left; vertical-align: top;
  }}
  th {{ color: var(--muted); font-weight: 500; position: sticky; top: 0; background: var(--bg); }}
  tbody tr:hover {{ background: #131a23; }}
  .sym {{ font-weight: 600; }}
  .score-hi {{ color: var(--hi); }}
  .score-md {{ color: var(--md); }}
  .score-lo {{ color: var(--lo); }}
  .muted {{ color: var(--muted); font-size: 11px; }}
  .alerts {{
    margin-top: 24px; padding: 16px; background: #11241a; border: 1px solid #1f3d2a;
    border-radius: 10px;
  }}
  .alerts h2 {{ margin: 0 0 8px 0; color: var(--hi); font-size: 16px; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>🤖 Toobit Scanner</h1>
    <div class="meta">Last run: {timestamp}</div>
  </div>
  <div>
    <span class="pill">Threshold: {threshold}</span>
    <span class="pill">Weights T:{weights.get('technical',0):.0f} P:{weights.get('pattern',0):.0f} S:{weights.get('social',0):.0f} W:{weights.get('whale',0):.0f}</span>
  </div>
</header>
<main>
  <div class="cards">
    <div class="card"><div class="label">Scanned</div><div class="value">{data.get('scanned', 0)}</div></div>
    <div class="card"><div class="label">Alerts (>{threshold})</div><div class="value" style="color: var(--hi)">{len(alerts)}</div></div>
    <div class="card"><div class="label">Avg score</div><div class="value">{(sum(r['score'] for r in data.get('results',[])) / max(1,len(data.get('results',[])))):.1f}</div></div>
    <div class="card"><div class="label">Top score</div><div class="value">{data.get('results',[{}])[0].get('score', 0):.1f}</div></div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Symbol</th><th>Score</th><th>Market cap</th><th>24h vol</th>
        <th>RSI / div</th><th>MACD / div</th><th>EMA</th>
        <th>Patterns</th><th>Social</th><th>Whale bias</th><th>24h %</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows) if rows else '<tr><td colspan="11">No data yet</td></tr>'}
    </tbody>
  </table>

  <div class="alerts">
    <h2>🔥 High-score alerts</h2>
    {"".join(f"<div>• <b>{a['symbol']}</b> — score {a['score']:.1f} (MCap ${a.get('market_cap_usd',0):,.0f})</div>" for a in alerts) or "<div>None.</div>"}
  </div>
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
