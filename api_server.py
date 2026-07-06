from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import random

app = FastAPI(title="SkyShield AI Backend API")

# Allow CORS for React frontend (Vite defaults to port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can restrict to ["http://localhost:5173"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load real model predictions if available
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FED_CSV = os.path.join(RESULTS_DIR, "federated_FD001_predictions.csv")

try:
    df_pred = pd.read_csv(FED_CSV)
except Exception:
    df_pred = pd.DataFrame({
        "y_true": np.random.randint(10, 130, 100),
        "y_pred": np.random.randint(10, 130, 100)
    })

AIRCRAFT_IDS = [
    "AX-101", "AX-102", "BX-214", "BX-215", "CX-319", 
    "CX-320", "DX-401", "DX-402", "EX-511", "EX-512"
]

def get_risk(rul):
    if rul < 25: return "Critical"
    if rul < 50: return "High"
    if rul < 80: return "Medium"
    return "Low"

def get_health(rul):
    return min(100, max(0, int((rul / 130) * 100)))

@app.get("/api/fleet")
def get_fleet():
    fleet = []
    # Sample from predictions for realistic RUL
    samples = df_pred.sample(len(AIRCRAFT_IDS)).to_dict('records')
    
    for i, a_id in enumerate(AIRCRAFT_IDS):
        rul = int(samples[i]["y_pred"])
        health = get_health(rul)
        risk = get_risk(rul)
        
        fleet.append({
            "id": a_id,
            "type": random.choice(["Boeing 737", "Boeing 777", "Airbus A320neo", "Airbus A350"]),
            "health": health,
            "rul": rul,
            "risk": risk,
            "phase": random.choice(["Cruise", "Takeoff", "Landing", "Descent", "Maintenance"]),
            "edgeStatus": random.choice(["Active", "Active", "Active", "Syncing", "Offline"]),
            "modelVersion": "v2.7.3",
            "maintenanceStatus": "OK" if risk in ["Low", "Medium"] else "Scheduled"
        })
    return fleet

@app.get("/api/aircraft/{aircraft_id}")
def get_aircraft(aircraft_id: str):
    rul = int(df_pred.sample(1).iloc[0]["y_pred"])
    return {
        "id": aircraft_id,
        "type": "Boeing 737",
        "health": get_health(rul),
        "rul": rul,
        "risk": get_risk(rul),
        "sensors": {
            "Fan Speed (N1)": 5200 + random.uniform(-50, 50),
            "Compressor Speed (N2)": 14800 + random.uniform(-100, 100),
            "Fuel Flow": 3820 + random.uniform(-40, 40),
            "Oil Pressure": 60 + random.uniform(-2, 2),
            "Oil Temp": 62 + random.uniform(-1, 1),
            "EGT": 1050 + random.uniform(-10, 10),
            "Vibration": 0.42 + random.uniform(-0.05, 0.05),
            "Pressure Ratio": 14.6 + random.uniform(-0.2, 0.2)
        }
    }

@app.get("/api/fl-status")
def get_fl_status():
    return {
        "currentRound": 24,
        "totalRounds": 50,
        "globalAccuracy": 94.2,
        "participants": 5,
        "encryption": "AES-256 GCM",
        "lastSync": "2 mins ago"
    }

@app.get("/api/edge-status")
def get_edge_status():
    return {
        "activeNodes": 412,
        "avgLatencyMs": 1.4,
        "totalPredictions": 1254300,
        "offlineCapable": True
    }

@app.get("/api/analytics")
def get_analytics():
    return {
        "accuracy": 94.2,
        "rmse": 14.5,
        "history": [
            {"round": i, "loss": 100 * (0.9 ** i) + random.uniform(0, 5)} 
            for i in range(1, 25)
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
