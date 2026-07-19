"""
Telegram notifier. Sends a single digest message and a message per
high-score coin.
"""
from __future__ import annotations
import os
import requests
from typing import List, Dict


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str, parse_mode: str = "HTML"):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.parse_mode = parse_mode
        self.base = f"https://api.telegram.org/bot{bot_token}"

    def _send(self, text: str, disable_web_preview: bool = True) -> bool:
        url = f"{self.base}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": self.parse_mode,
            "disable_web_page_preview": disable_web_preview,
        }
        try:
            r = requests.post(url, json=payload, timeout=15)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def send_digest(self, total_scanned: int, alerts: List[Dict], weights: Dict[str, float]) -> bool:
        """Send a single summary message."""
        ts = alerts[0].get("timestamp", "") if alerts else ""
        lines = [
            "🤖 <b>Toobit Scanner — 12h Report</b>",
            f"🕒 {ts}",
            f"📊 Symbols scanned: <b>{total_scanned}</b>",
            f"🔥 High-score alerts (>85): <b>{len(alerts)}</b>",
            f"⚖️ Weights: T={weights.get('technical', 0):.1f} "
            f"P={weights.get('pattern', 0):.1f} "
            f"S={weights.get('social', 0):.1f} "
            f"W={weights.get('whale', 0):.1f}",
            "",
        ]
        if alerts:
            for a in alerts[:5]:
                lines.append(self._format_alert(a))
        else:
            lines.append("✅ No high-confidence setups in this run.")
        return self._send("\n".join(lines))

    def send_alert(self, alert: Dict) -> bool:
        return self._send(self._format_alert(alert, header=True))

    def _format_alert(self, a: Dict, header: bool = False) -> str:
        div_rsi = a.get("rsi_divergence", "none")
        div_macd = a.get("macd_divergence", "none")
        ema_align = a.get("ema_alignment", "mixed")
        patterns = ", ".join(a.get("patterns", [])) or "—"
        emoji = "🚨" if a["score"] >= 90 else "🔥"
        title = f"{emoji} <b>{a['symbol']}</b> — Score <b>{a['score']:.1f}</b>\n" if header else \
                f"• <b>{a['symbol']}</b> — <b>{a['score']:.1f}</b> | "
        return (
            title
            + f"MCap ${a.get('market_cap_usd', 0):,.0f} | "
              f"24h vol ${a.get('quote_volume_24h', 0):,.0f}\n"
            + f"  RSI={a.get('rsi_value', 0):.1f} ({div_rsi}) | "
              f"MACD hist={a.get('macd_hist', 0):.4f} ({div_macd})\n"
            + f"  EMA: {ema_align} | Patterns: {patterns}\n"
            + f"  Social={a.get('social_score', 0):.0f} | "
              f"Whale bias={a.get('liq_bias', 0):+.2f}\n"
        )

    @classmethod
    def from_env(cls):
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat:
            return None
        return cls(token, chat)
