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
    "AKEUSDT": {
        "name": "Akedo",
        "pump_count_30d": 3,
        "avg_gain": 97.4,
        "last_pump": "2026-07-17T20:00:00+00:00",
        "volatility": "VERY_HIGH",
        # AKE: very high vol, very fast moves. Wider TP, tighter SL.
        "tp_pct": 10.0,
        "sl_pct": 2.0,
        "trail_pct": 3.5,
        "max_hold_hours": 4.0,
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
        # IKA: highest single gain. Aggressive params.
        "tp_pct": 12.0,
        "sl_pct": 2.5,
        "trail_pct": 4.0,
        "max_hold_hours": 6.0,
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
}

# Secondary watchlist: 2+ pumps but lower confidence (DN demoted here)
SECONDARY_WATCHLIST = {
    "DNUSDT",  # demoted: vol 24h too low ($140K) for primary watch
    "CHECKUSDT", "ZEUSUSDT", "CAMPUSDT", "ARGUSDT", "VELVETUSDT",
    "INUSDT", "OXTUSDT", "IDOLUSDT", "SAROSUSDT", "EGL1USDT", "UNIONUSDT"
}

# Cluster follow settings: when a pump is detected, watch all repeaters for 24h
CLUSTER_FOLLOW_HOURS = 24

# Two-stage entry: how much size to put on pre-pump signal
PRE_PUMP_SIZE_FRACTION = 0.30  # 30% at pre-pump (t-1h)
CONFIRM_SIZE_FRACTION = 0.70   # 70% at confirmation (t+0)

# Confidence thresholds
PRE_PUMP_CONFIDENCE_MIN = 50.0  # Min conf to enter with 30% size
CONFIRM_CONFIDENCE_MIN = 60.0   # Min conf to add 70% more
