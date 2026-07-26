#!/bin/bash
# Simple collector runner
cd /home/user/toobit-scanner
while true; do
    echo "[$(date -u +%H:%M:%S)] Starting cycle..." >> /tmp/pumphunter.log
    timeout 500 python3 -m src.live_collector --once >> /tmp/pumphunter.log 2>&1
    rc=$?
    echo "[$(date -u +%H:%M:%S)] Done rc=$rc, sleeping 10min..." >> /tmp/pumphunter.log
    sleep 600
done
