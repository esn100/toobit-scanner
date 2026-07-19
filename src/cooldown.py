"""
Cooldown / Duplicate Signal Guard (Layer 1 of the PumpHunter pipeline).

Prevents spamming signals for the same (exchange, symbol) within a
configurable cooldown window. Persists state to a JSON file so it
survives across CI runs.
"""
from __future__ import annotations
import os
import json
import time
from typing import Dict, Optional


class CooldownGuard:
    def __init__(self, path: str, default_hours: float = 6.0):
        self.path = path
        self.default_hours = default_hours
        self._state: Dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._state = json.load(f)
            except Exception:
                self._state = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2)

    def is_cool(self, key: str, hours: Optional[float] = None) -> bool:
        """Return True if `key` is allowed to fire a new signal."""
        h = hours or self.default_hours
        last = self._state.get(key, 0.0)
        return (time.time() - last) >= h * 3600.0

    def mark(self, key: str) -> None:
        self._state[key] = time.time()
        self._save()

    def clear(self, key: str) -> None:
        self._state.pop(key, None)
        self._save()

    def age_hours(self, key: str) -> float:
        last = self._state.get(key, 0.0)
        if last == 0.0:
            return float("inf")
        return (time.time() - last) / 3600.0
