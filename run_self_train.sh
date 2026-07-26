#!/bin/bash
# Self-training loop - runs every hour
cd /home/user/toobit-scanner
while true; do
    echo "[$(date -u +%H:%M:%S)] Starting self_train iteration..." >> /tmp/self_train.log
    python3 -m src.self_train --once >> /tmp/self_train.log 2>&1
    echo "[$(date -u +%H:%M:%S)] Sleeping 1 hour..." >> /tmp/self_train.log
    sleep 3600
done
