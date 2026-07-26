#!/bin/bash
# PumpHunter-AI Dual Picks - quick launcher
# Always shows best LONG and SHORT together from 77-symbol watchlist
cd /home/user/toobit-scanner
echo "🔍 PumpHunter-AI Dual Picks (LONG + SHORT)"
echo ""
python3 -m src.live_dual_picks --top 8 --min-score 25
