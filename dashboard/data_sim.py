"""
dashboard/data_sim.py
---------------------
Simulated live data engine for Maverick AI dashboard.
All sensor values, fleet state, and federated metrics are generated
realistically so the dashboard works without a live backend.
"""

import numpy as np
import random
import time
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

AIRCRAFT_IDS = ["AX-101", "AX-102", "BX-214", "BX-215", "CX-319", "CX-320", "DX-401", "DX-402", "EX-511", "EX-512"]
AIRCRAFT_TYPES = {
    "AX": "Boeing 737-800", "BX": "Airbus A320neo",
    "CX": "Boeing 777-300", "DX": "Airbus A350-900", "EX": "Boeing 787-9"
}
FLIGHT_PHASES = ["Cruise", "Takeoff", "Climb", "Descent", "Landing", "Ground", "Maintenance"]
AIRLINES = ["Airline Alpha", "Airline Beta", "Airline Gamma", "Airline Delta", "Airline Epsilon"]

def get_aircraft_type(aid):
    prefix = aid[:2]
    return AIRCRAFT_TYPES.get(prefix, "Unknown")

def make_fleet(seed_offset=0):
    fleet = []
    for i, aid in enumerate(AIRCRAFT_IDS):
        r = random.Random(i + seed_offset)
        health = r.uniform(45, 99)
        rul = int(r.uniform(8, 125))
        if rul < 20: risk = "Critical"
        elif rul < 40: risk = "High"
        elif rul < 70: risk = "Medium"
        else: risk = "Low"
        fleet.append({
            "Aircraft ID": aid,
            "Type": get_aircraft_type(aid),
            "Health %": round(health, 1),
            "RUL (cycles)": rul,
            "Risk": risk,
            "Flight Phase": r.choice(FLIGHT_PHASES),
            "Edge AI": "Active" if r.random() > 0.1 else "Offline",
            "Last FL Update": f"{r.randint(1,48)}h ago",
            "Maintenance": "Scheduled" if risk in ["High","Critical"] else "OK",
        })
    return fleet

def get_sensor_values(base_seed=None):
    """Return a dict of current simulated sensor readings."""
    t = time.time() if base_seed is None else base_seed
    noise = lambda mu, sigma: float(np.random.normal(mu, sigma))
    return {
        "Fan Speed (N1)":        round(noise(5200 + 200 * np.sin(t * 0.1), 30), 1),
        "Compressor Speed (N2)": round(noise(14800 + 100 * np.sin(t * 0.07), 50), 1),
        "Fuel Flow (Wf)":        round(noise(3820 + 80 * np.sin(t * 0.05), 20), 1),
        "Oil Pressure":          round(noise(60 + 2 * np.sin(t * 0.2), 0.5), 2),
        "Oil Temperature":       round(noise(62 + 3 * np.sin(t * 0.08), 0.8), 1),
        "Exhaust Gas Temp":      round(noise(1050 + 20 * np.sin(t * 0.06), 5), 1),
        "Vibration":             round(noise(0.42 + 0.05 * np.sin(t * 0.3), 0.01), 3),
        "Pressure Ratio":        round(noise(14.6 + 0.2 * np.sin(t * 0.04), 0.05), 2),
        "Ambient Temp":          round(noise(-28 + 2 * np.sin(t * 0.02), 0.3), 1),
        "Altitude (ft)":         round(noise(35000 + 500 * np.sin(t * 0.01), 100), 0),
        "Flight Cycles":         int(noise(1420, 5)),
        "Flight Hours":          round(noise(3840 + 0.5, 2), 1),
    }

def get_feature_importance():
    sensors = [
        "Compressor Temp (T48)", "Fuel Flow (Wf)", "Pressure Ratio (P30)",
        "N2 Speed", "Exhaust Gas Temp", "Fan Speed (N1)",
        "Oil Temperature", "Vibration", "Oil Pressure"
    ]
    weights = sorted(np.random.dirichlet(np.ones(len(sensors))) * 100, reverse=True)
    return [{"sensor": s, "importance": round(w, 2)} for s, w in zip(sensors, weights)]

def get_rul_prediction(aircraft_id=None):
    seed = hash(aircraft_id) % 100 if aircraft_id else 42
    base = [125, 42, 87, 60, 18, 110, 34, 95, 22, 78]
    idx = AIRCRAFT_IDS.index(aircraft_id) if aircraft_id in AIRCRAFT_IDS else 0
    return base[idx]

def get_health_score(aircraft_id=None):
    base = [92, 61, 78, 55, 33, 88, 49, 82, 38, 71]
    idx = AIRCRAFT_IDS.index(aircraft_id) if aircraft_id in AIRCRAFT_IDS else 0
    return base[idx]

def get_federated_clients():
    clients = []
    for i, name in enumerate(AIRLINES):
        r = random.Random(i)
        clients.append({
            "airline": name,
            "status": "Active" if r.random() > 0.1 else "Syncing",
            "local_training": r.choice(["Idle", "Training", "Complete"]),
            "model_version": f"v2.{r.randint(3,8)}.{r.randint(0,9)}",
            "last_sync": f"{r.randint(1,60)} min ago",
            "privacy": "Protected",
            "connection": r.choice(["Excellent","Good","Good","Fair"]),
            "raw_data_shared": "NO",
            "params_shared": "YES",
            "local_engines": r.randint(12, 45),
        })
    return clients

def get_alerts():
    return [
        {"id":"AX-102","level":"Critical","msg":"RUL below threshold — inspect within 8 cycles","time":"2 min ago","icon":"🔴"},
        {"id":"BX-214","level":"High","msg":"High Compressor Degradation detected","time":"14 min ago","icon":"🟠"},
        {"id":"CX-319","level":"High","msg":"Elevated Vibration — Sensor s8 anomaly","time":"31 min ago","icon":"🟠"},
        {"id":"DX-401","level":"Medium","msg":"Fuel Flow deviation — monitor closely","time":"1h ago","icon":"🟡"},
        {"id":"EX-511","level":"Medium","msg":"Oil Temperature trending upward","time":"2h ago","icon":"🟡"},
        {"id":"AX-101","level":"Low","msg":"Scheduled FL model sync completed","time":"3h ago","icon":"🟢"},
    ]

def get_maintenance_recommendations(aircraft_id=None):
    return [
        {
            "action": "Inspect High Pressure Compressor (HPC)",
            "priority": "URGENT",
            "window": "Next 5 cycles",
            "downtime": "~6 hours",
            "technician": "Level 3 Certified",
            "parts": ["HPC Blade Set", "Compressor Seal Kit"],
            "confidence": "94%",
        },
        {
            "action": "Replace Oil Filter & Check Oil Pressure Lines",
            "priority": "HIGH",
            "window": "Next 15 cycles",
            "downtime": "~2 hours",
            "technician": "Level 2 Certified",
            "parts": ["Oil Filter x2", "Pressure Sensor"],
            "confidence": "88%",
        },
        {
            "action": "Borescope Inspection — Turbine Blades",
            "priority": "MEDIUM",
            "window": "Next 30 cycles",
            "downtime": "~4 hours",
            "technician": "Level 3 Certified",
            "parts": ["Inspection Kit"],
            "confidence": "76%",
        },
    ]

def get_fl_training_history():
    rounds = list(range(1, 21))
    loss = [4500 * np.exp(-0.25 * r) + 140 + np.random.normal(0, 15) for r in rounds]
    rmse = [73 * np.exp(-0.2 * r) + 12.3 + np.random.normal(0, 0.5) for r in rounds]
    return {"rounds": rounds, "loss": loss, "rmse": rmse}

def get_edge_node_stats():
    return {
        "cpu": round(np.random.uniform(18, 42), 1),
        "memory": round(np.random.uniform(34, 58), 1),
        "latency_ms": round(np.random.uniform(2.1, 6.8), 1),
        "predictions_today": int(np.random.uniform(1200, 2800)),
        "model_version": "v2.7.3",
        "offline_capable": True,
        "inference_location": "Airline Edge Server",
        "uptime": "99.97%",
    }

def get_health_trend():
    days = list(range(30, 0, -1))
    scores = [85 - i * 0.4 + np.random.normal(0, 1.5) for i in range(30)]
    ruls = [80 - i * 1.2 + np.random.normal(0, 2) for i in range(30)]
    return {"days_ago": days, "health": scores, "rul": ruls}
