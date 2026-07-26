"""
Repeater configuration: which symbols to watch 24/7.
Derived from analysis of 49 historical pumps on small caps.
6 symbols account for 25%+ of all detected pumps.
"""
REPEATERS = {
    "EVAAUSDT": {
        "name": "EVAA Protocol",
        "pump_count_30d": 4,
        "avg_gain": 155.5,
        "last_pump": "2026-07-13T00:00:00+00:00",
        "volatility": "HIGH",
        # Optimal TP/SL for this repeater (from data)
        "tp_pct": 8.0,
        "sl_pct": 2.5,
        "trail_pct": 3.0,
        "max_hold_hours": 8.0,
        # Pattern match thresholds
        "pre_pump_rvol_min": 0.5,
        "pre_pump_max_rvol_4h_min": 1.5,
        "pre_pump_mom_3_min": -5.0,
        "pre_pump_mom_3_max": 5.0,
        "pre_pump_flat_min_hours": 3.0,
    },
    "TLMUSDT": {
        "name": "Alien Worlds Trilium",
        "pump_count_30d": 4,
        "avg_gain": 102.8,
        "last_pump": "2026-07-18T20:00:00+00:00",
        "volatility": "MEDIUM",
        "strategy": "pump_runner",  # 4 historical pumps, multi-wave
        "tp_pct": 20.0,
        "sl_pct": 2.5,
        "trail_pct": 5.0,
        "trail_activate_pct": 15.0,
        "max_hold_hours": 12.0,
        "reentry_on_dip": True,
        "reentry_rsi_max": 35.0,
        "reentry_size": 0.30,
        "pre_pump_rvol_min": 0.5,
        "pre_pump_max_rvol_4h_min": 1.3,
        "pre_pump_mom_3_min": -3.0,
        "pre_pump_mom_3_max": 3.0,
        "pre_pump_flat_min_hours": 2.0,
    },
    "LABUSDT": {
        "name": "Lab",
        "pump_count_30d": 3,
        "avg_gain": 108.0,
        "last_pump": "2026-07-13T13:00:00+00:00",
        "volatility": "HIGH",
        "tp_pct": 8.0,
        "sl_pct": 2.5,
        "trail_pct": 3.0,
        "max_hold_hours": 6.0,
        "pre_pump_rvol_min": 0.5,
        "pre_pump_max_rvol_4h_min": 1.5,
        "pre_pump_mom_3_min": -5.0,
        "pre_pump_mom_3_max": 5.0,
        "pre_pump_flat_min_hours": 2.0,
    },
    "BANKUSDT": {
        "name": "Bank",
        "pump_count_30d": 3,
        "avg_gain": 97.8,
        "last_pump": "2026-07-22T12:00:00+00:00",
        "volatility": "MEDIUM",
        "strategy": "pump_runner",
        "tp_pct": 20.0,
        "sl_pct": 2.5,
        "trail_pct": 5.0,
        "trail_activate_pct": 15.0,
        "max_hold_hours": 12.0,
        "reentry_on_dip": True,
        "reentry_rsi_max": 35.0,
        "reentry_size": 0.30,
        "pre_pump_rvol_min": 0.5,
        "pre_pump_max_rvol_4h_min": 1.3,
        "pre_pump_mom_3_min": -3.0,
        "pre_pump_mom_3_max": 3.0,
        "pre_pump_flat_min_hours": 2.0,
    },
    "AKEUSDT": {
        "name": "Akedo",
        "pump_count_30d": 3,
        "avg_gain": 97.4,
        "last_pump": "2026-07-17T20:00:00+00:00",
        "volatility": "VERY_HIGH",
        "strategy": "pump_runner",  # 3 pumps avg +97%, fast mover
        "tp_pct": 25.0,
        "sl_pct": 2.5,
        "trail_pct": 7.0,
        "trail_activate_pct": 15.0,
        "max_hold_hours": 12.0,
        "reentry_on_dip": True,
        "reentry_rsi_max": 30.0,
        "reentry_size": 0.30,
        "pre_pump_rvol_min": 0.8,
        "pre_pump_max_rvol_4h_min": 2.0,  # needs more vol to be valid
        "pre_pump_mom_3_min": -10.0,  # AKE has violent pullbacks
        "pre_pump_mom_3_max": 10.0,
        "pre_pump_flat_min_hours": 1.0,  # AKE pumps come fast
    },
    "IKAUSDT": {
        "name": "Ika",
        "pump_count_30d": 1,
        "avg_gain": 235.2,
        "last_pump": "2026-07-01T13:00:00+00:00",
        "volatility": "VERY_HIGH",
        "strategy": "pump_runner",  # biggest single pump, let it run
        "tp_pct": 30.0,
        "sl_pct": 3.0,
        "trail_pct": 10.0,
        "trail_activate_pct": 25.0,
        "max_hold_hours": 24.0,
        "reentry_on_dip": True,
        "reentry_rsi_max": 35.0,
        "reentry_size": 0.30,
        "pre_pump_rvol_min": 0.8,
        "pre_pump_max_rvol_4h_min": 2.0,
        "pre_pump_mom_3_min": -8.0,
        "pre_pump_mom_3_max": 8.0,
        "pre_pump_flat_min_hours": 1.0,
    },
    "SYNUSDT": {
        "name": "Synapse",
        "pump_count_30d": 2,
        "avg_gain": 102.7,
        "last_pump": "2026-06-24T13:00:00+00:00",
        "volatility": "HIGH",
        "strategy": "pump_runner",  # multi-leg runner strategy
        "tp_pct": 30.0,        # 30% TP (SYN pumps 30-50% typically)
        "sl_pct": 3.0,         # 3% SL
        "trail_pct": 8.0,      # loose trail after +20%
        "trail_activate_pct": 20.0,  # start trailing after +20%
        "max_hold_hours": 24.0,  # long hold for multi-wave
        "reentry_on_dip": True,    # add more on RSI<35
        "reentry_rsi_max": 35.0,
        "reentry_size": 0.30,      # add 30% on dip
        "pre_pump_rvol_min": 0.5,
        "pre_pump_max_rvol_4h_min": 1.5,
        "pre_pump_mom_3_min": -5.0,
        "pre_pump_mom_3_max": 5.0,
        "pre_pump_flat_min_hours": 2.0,
    },
    "ACEUSDT": {
        "name": "Fusionist ACE",
        "pump_count_30d": 1,
        "avg_gain": 96.4,
        "last_pump": "2026-07-19T06:00:00+00:00",
        "volatility": "HIGH",
        "strategy": "pump_runner",
        "tp_pct": 20.0,
        "sl_pct": 2.5,
        "trail_pct": 5.0,
        "trail_activate_pct": 15.0,
        "max_hold_hours": 12.0,
        "reentry_on_dip": True,
        "reentry_rsi_max": 35.0,
        "reentry_size": 0.30,
        "pre_pump_rvol_min": 0.5,
        "pre_pump_max_rvol_4h_min": 1.5,
        "pre_pump_mom_3_min": -5.0,
        "pre_pump_mom_3_max": 5.0,
        "pre_pump_flat_min_hours": 2.0,
    },
    "INUSDT": {
        "name": "IN",
        "pump_count_30d": 1,
        "avg_gain": 111.0,
        "last_pump": "2026-06-29T14:00:00+00:00",
        "volatility": "HIGH",
        "strategy": "pump_runner",
        "tp_pct": 20.0,
        "sl_pct": 2.5,
        "trail_pct": 5.0,
        "trail_activate_pct": 15.0,
        "max_hold_hours": 12.0,
        "reentry_on_dip": True,
        "reentry_rsi_max": 35.0,
        "reentry_size": 0.30,
        "pre_pump_rvol_min": 0.5,
        "pre_pump_max_rvol_4h_min": 1.5,
        "pre_pump_mom_3_min": -5.0,
        "pre_pump_mom_3_max": 5.0,
        "pre_pump_flat_min_hours": 2.0,
    },
    "ERAUSDT": {
        "name": "Era",
        "pump_count_30d": 1,
        "avg_gain": 83.8,
        "last_pump": "2026-07-20T06:00:00+00:00",
        "volatility": "MEDIUM",
        "tp_pct": 7.0,
        "sl_pct": 2.0,
        "trail_pct": 2.5,
        "max_hold_hours": 6.0,
        "pre_pump_rvol_min": 0.5,
        "pre_pump_max_rvol_4h_min": 1.3,
        "pre_pump_mom_3_min": -3.0,
        "pre_pump_mom_3_max": 3.0,
        "pre_pump_flat_min_hours": 2.0,
    },
    "WODUSDT": {
        "name": "World of Dypians",
        "pump_count_30d": 1,
        "avg_gain": 111.9,
        "last_pump": "2026-06-26T14:00:00+00:00",
        "volatility": "HIGH",
        "strategy": "pump_runner",
        "tp_pct": 20.0,
        "sl_pct": 2.5,
        "trail_pct": 5.0,
        "trail_activate_pct": 15.0,
        "max_hold_hours": 12.0,
        "reentry_on_dip": True,
        "reentry_rsi_max": 35.0,
        "reentry_size": 0.30,
        "pre_pump_rvol_min": 0.5,
        "pre_pump_max_rvol_4h_min": 1.5,
        "pre_pump_mom_3_min": -5.0,
        "pre_pump_mom_3_max": 5.0,
        "pre_pump_flat_min_hours": 2.0,
    },
    "XCXUSDT": {
        "name": "XCX",
        "pump_count_30d": 1,
        "avg_gain": 161.6,
        "last_pump": "2026-06-25T07:00:00+00:00",
        "volatility": "VERY_HIGH",
        "strategy": "pump_runner",  # biggest gain, runner strategy
        "tp_pct": 30.0,
        "sl_pct": 3.0,
        "trail_pct": 8.0,
        "trail_activate_pct": 20.0,
        "max_hold_hours": 24.0,
        "reentry_on_dip": True,
        "reentry_rsi_max": 35.0,
        "reentry_size": 0.30,
        "pre_pump_rvol_min": 0.8,
        "pre_pump_max_rvol_4h_min": 1.8,
        "pre_pump_mom_3_min": -7.0,
        "pre_pump_mom_3_max": 7.0,
        "pre_pump_flat_min_hours": 1.5,
    },
}

# Secondary watchlist: 2+ pumps but lower confidence
SECONDARY_WATCHLIST = {
    "DNUSDT",  # demoted: vol 24h too low ($140K) for primary watch
    "ARGUSDT", "ZEUSUSDT", "CAMPUSDT", "VELVETUSDT", "CHECKUSDT",
    "BREVUSDT", "BIRBUSDT", "RAVEUSDT", "IMUUSDT", "IDOLUSDT",
    "VOOIUSDT", "AIGENSYNUSDT", "NOMUSDT", "SAROSUSDT", "UNIONUSDT",
    "OXTUSDT", "EGL1USDT",
    # === CAP-BASED WATCHLIST (always monitored) ===
    # LOW CAP (<$50M) - highest pump potential
    "TUTUSDT",   # Tutorial - $13.4M
    "WCTUSDT",   # WalletConnect - $17.4M
    "EPICUSDT",  # Epic Chain - $25.0M
    "BRUSDT",    # Bedrock - $44.1M
    # MID CAP ($50M-$500M) - good balance
    "EULUSDT",     # Euler - $64.9M
    "PIEVERSEUSDT",# Pieverse - $65.2M
    "LPTUSDT",     # Livepeer - $74.5M
    "ESPUSDT",     # Espresso - $108.8M
    "FARTCOINUSDT",# Fartcoin - $129.1M
    "SENTUSDT",    # Sentient - $160.6M
    "ZAMAUSDT",    # Zama - $175.6M
    "KAITOUSDT",   # Kaito - $283.6M
    "VIRTUALUSDT", # Virtuals Protocol - $387.0M
    "TRUMPUSDT",   # Official Trump - $395.4M
    "AEROUSDT",    # Aerodrome - $402.9M
    "INJUSDT",     # Injective Protocol - $487.5M
    "BANKUSDT",    # Lorenzo Governance - $478.4M
    # === CAP-BASED WATCHLIST (auto-imported from CSV) ===
    # LOW CAP (<$50M) - highest pump potential
    "KILOUSDT",  # KiloEx Token - $2.1M
    "AFCUSDT",  # Arsenal Fan Token - $3.7M
    "ACEUSDT",  # Fusionist - $8.6M
    "RESOLVUSDT",  # Resolv - $8.6M
    "BERTUSDT",  # Bertram The Pomeranian - $9.3M
    "OPNUSDT",  # Opinion - $9.4M
    "FHEUSDT",  # Mind Network - $10.5M
    "TLMUSDT",  # Alien Worlds - $12.2M
    "TUTUSDT",  # Tutorial - $13.5M
    "PARTIUSDT",  # Particle Network - $13.5M
    "WCTUSDT",  # WalletConnect - $17.4M
    "BREVUSDT",  # Brevis Token - $19.1M
    "ALCHUSDT",  # Alchemist AI - $21.8M
    "EPICUSDT",  # Epic Chain - $24.9M
    "LAUSDT",  # Lagrange - $25.4M
    "HOLOUSDT",  # Holoworld AI - $29.4M
    "SYNUSDT",  # Synapse - $31.7M
    "CAPUSDT",  # Cap - $34.1M
    "ZEREBROUSDT",  # zerebro - $36.4M
    "ATUSDT",  # APRO oracle Token - $36.5M
    "BOMEUSDT",  # BOOK OF MEME - $36.5M
    "MOODENGUSDT",  # Moo Deng - $37.5M
    "BRUSDT",  # Bedrock - $44.0M
    "NESUSDT",  # Nesa - $44.4M
    "BABYUSDT",  # Babylon - $49.9M
    # MID CAP ($50M-$500M) - good balance
    "PIEVERSEUSDT",  # Pieverse Token - $65.1M
    "EULUSDT",  # Euler - $66.5M
    "RAVEUSDT",  # RaveDAO - $69.8M
    "HUMAUSDT",  # Huma Finance - $70.4M
    "LPTUSDT",  # Livepeer - $75.1M
    "NXPCUSDT",  # NEXPACE - $75.1M
    "ORDIUSDT",  # ordi - $79.9M
    "ALLOUSDT",  # Allora - $80.1M
    "ORCAUSDT",  # Orca - $89.2M
    "KMNOUSDT",  # Kamino - $92.7M
    "ESPUSDT",  # Espresso - $101.5M
    "FARTCOINUSDT",  # Fartcoin - $129.0M
    "SENTUSDT",  # Sentient - $160.4M
    "ZAMAUSDT",  # Zama - $174.3M
    "GRASSUSDT",  # Grass - $215.5M
    "REUSDT",  # Re Protocol - $225.9M
    "BONKUSDT",  # Bonk - $270.7M
    "KAITOUSDT",  # Kaito - $281.8M
    "JTOUSDT",  # JITO - $293.8M
    "LDOUSDT",  # Lido DAO Token - $313.1M
    "FDUSDUSDT",  # First Digital USD - $346.5M
    "FETUSDT",  # Artificial Superintelligence Alliance - $368.5M
    "VIRTUALUSDT",  # Virtuals Protocol - $387.2M
    "TRUMPUSDT",  # OFFICIAL TRUMP - $394.1M
    "PENGUUSDT",  # Pudgy Penguins - $396.2M
    "AEROUSDT",  # Aerodrome - $403.3M
    "UBUSDT",  # Unibase - $413.1M
    "DASHUSDT",  # Dash - $416.9M
    "BANKUSDT",  # Lorenzo Governance Token - $481.6M
    "INJUSDT",  # Injective Protocol - $487.3M
}


# Cluster follow settings: when a pump is detected, watch all repeaters for 24h
CLUSTER_FOLLOW_HOURS = 24

# Two-stage entry: how much size to put on pre-pump signal
PRE_PUMP_SIZE_FRACTION = 0.30  # 30% at pre-pump (t-1h)
CONFIRM_SIZE_FRACTION = 0.70   # 70% at confirmation (t+0)

# Confidence thresholds
# Based on backtest of 9 signals: conf 75-85% = 100% WR
# Higher conf (95%+) actually WORSE due to overfitting on edge cases
PRE_PUMP_CONFIDENCE_MIN = 50.0  # Min conf to enter with 30% size
CONFIRM_CONFIDENCE_MIN = 60.0   # Min conf to add 70% more
# Cap confidence to prevent overconfidence
# (a 100% conf signal was actually worse than 75-85%)
MAX_USABLE_CONFIDENCE = 85.0    # Cap at 85% even if pattern scores higher
