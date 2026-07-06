"""dashboard/pages_aircraft.py — Aircraft Monitor & Sensor pages."""
import streamlit as st
import time, random
import pandas as pd
from dashboard.data_sim import (AIRCRAFT_IDS, get_aircraft_type, get_sensor_values,
    get_rul_prediction, get_health_score, get_feature_importance,
    get_maintenance_recommendations, get_health_trend)
from dashboard.styles import page_header, kpi_card
from dashboard.charts import rul_gauge, health_gauge, feature_importance_chart, health_trend_chart


AIRCRAFT_SILHOUETTE = """
<div style="background:rgba(0,20,40,0.7);border:1px solid rgba(0,191,255,0.2);
    border-radius:14px;padding:20px;text-align:center;position:relative;overflow:hidden">
    <div style="position:absolute;top:0;left:0;right:0;bottom:0;
        background:radial-gradient(ellipse at center, rgba(0,80,160,0.15) 0%, transparent 70%)"></div>
    <div style="font-size:5rem;filter:drop-shadow(0 0 20px rgba(0,191,255,0.6))">✈</div>
    <div style="margin-top:8px">
        <span style="display:inline-block;background:{eng1_color};width:12px;height:12px;
            border-radius:50%;margin:0 4px;box-shadow:0 0 8px {eng1_color}"></span>
        <span style="color:#6B8CAE;font-size:0.72rem">ENGINE 1</span>
        <span style="display:inline-block;background:{eng2_color};width:12px;height:12px;
            border-radius:50%;margin:0 4px 0 16px;box-shadow:0 0 8px {eng2_color}"></span>
        <span style="color:#6B8CAE;font-size:0.72rem">ENGINE 2</span>
    </div>
</div>
"""


def render_aircraft_monitor():
    st.markdown(page_header("🔍", "Aircraft Monitor",
        "Individual aircraft diagnostics, sensor data, and AI predictions"), unsafe_allow_html=True)

    selected = st.selectbox("Select Aircraft", AIRCRAFT_IDS, key="ac_select")
    rul = get_rul_prediction(selected)
    health = get_health_score(selected)
    sensors = get_sensor_values()
    feat_imp = get_feature_importance()
    recs = get_maintenance_recommendations(selected)
    trend = get_health_trend()

    # Engine colors based on RUL
    e1_color = "#00E676" if rul > 60 else ("#FFB300" if rul > 25 else "#FF4444")
    e2_color = "#00E676" if health > 70 else ("#FFB300" if health > 40 else "#FF4444")

    st.markdown("---")
    top_l, top_r = st.columns([1, 2])

    with top_l:
        silhouette = AIRCRAFT_SILHOUETTE.replace("{eng1_color}", e1_color).replace("{eng2_color}", e2_color)
        st.markdown(silhouette, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="margin-top:10px;background:rgba(0,20,40,0.6);border:1px solid rgba(0,191,255,0.15);
            border-radius:10px;padding:12px 16px">
            <div style="font-size:0.72rem;color:#6B8CAE;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.06em">Aircraft Info</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:0.78rem">
                <div><span style="color:#6B8CAE">ID:</span> <span style="color:#E8F4FF;font-weight:600">{selected}</span></div>
                <div><span style="color:#6B8CAE">Type:</span> <span style="color:#E8F4FF">{get_aircraft_type(selected)}</span></div>
                <div><span style="color:#6B8CAE">Cycles:</span> <span style="color:#E8F4FF">{sensors['Flight Cycles']}</span></div>
                <div><span style="color:#6B8CAE">Hours:</span> <span style="color:#E8F4FF">{sensors['Flight Hours']}h</span></div>
                <div><span style="color:#6B8CAE">Altitude:</span> <span style="color:#E8F4FF">{int(sensors['Altitude (ft)']):,} ft</span></div>
                <div><span style="color:#6B8CAE">Phase:</span> <span style="color:#00E676">Cruise</span></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with top_r:
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(rul_gauge(rul), use_container_width=True)
        with g2:
            st.plotly_chart(health_gauge(health), use_container_width=True)

    st.markdown("---")

    # ── Sensor Grid ─────────────────────────────────────────────────────────
    st.markdown("#### 📡 Live Sensor Readings")
    sensor_items = list(sensors.items())[:10]
    s_cols = st.columns(5)
    sensor_colors = ["#00BFFF","#00E676","#FFB300","#9B59B6","#FF6B6B",
                     "#00BFFF","#00E676","#FFB300","#9B59B6","#FF6B6B"]
    for i, ((k, v), color) in enumerate(zip(sensor_items, sensor_colors)):
        with s_cols[i % 5]:
            st.markdown(f"""
            <div style="background:rgba(0,20,40,0.7);border:1px solid rgba(0,191,255,0.12);
                border-radius:10px;padding:12px 10px;margin-bottom:8px;
                border-top:2px solid {color}">
                <div style="font-size:0.68rem;color:#6B8CAE;margin-bottom:4px">{k}</div>
                <div style="font-size:1.2rem;font-weight:700;color:{color}">{v}</div>
                <div style="height:3px;background:rgba(255,255,255,0.08);border-radius:2px;margin-top:6px">
                    <div style="height:3px;background:{color};border-radius:2px;width:{random.randint(40,90)}%;
                        animation: pulse 2s infinite"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── XAI + Maintenance ────────────────────────────────────────────────────
    xai_col, maint_col = st.columns([1, 1])

    with xai_col:
        st.markdown("#### 🧠 Explainable AI — Why this prediction?")
        st.markdown(f"""
        <div style="background:rgba(0,40,80,0.5);border:1px solid rgba(0,191,255,0.2);
            border-radius:10px;padding:14px 16px;margin-bottom:10px;font-size:0.82rem;color:#C8D8E8;
            border-left:3px solid #00BFFF">
            <b style="color:#00BFFF">AI Explanation:</b> The RUL prediction of <b style="color:#FFB300">{rul} cycles</b>
            was primarily driven by degradation patterns in <b>Compressor Temperature</b>,
            <b>Fuel Flow</b>, and <b>Pressure Ratio</b>. The model detected a progressive
            increase in T48 over the last 30 cycles consistent with HPC deterioration.
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(feature_importance_chart(feat_imp), use_container_width=True)

    with maint_col:
        st.markdown("#### 🔧 Maintenance Recommendations")
        priority_colors = {"URGENT":"#FF0055","HIGH":"#FF4444","MEDIUM":"#FFB300","LOW":"#00E676"}
        for rec in recs:
            color = priority_colors.get(rec["priority"], "#888")
            st.markdown(f"""
            <div style="background:rgba(0,20,40,0.7);border:1px solid rgba(0,191,255,0.15);
                border-left:3px solid {color};border-radius:0 10px 10px 0;
                padding:12px 14px;margin-bottom:8px">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                    <span style="color:#E8F4FF;font-weight:600;font-size:0.82rem">{rec['action']}</span>
                    <span style="background:{color}25;color:{color};padding:2px 8px;
                        border-radius:20px;font-size:0.68rem;font-weight:700">{rec['priority']}</span>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:0.72rem;color:#6B8CAE">
                    <span>⏱ Window: <b style="color:#C8D8E8">{rec['window']}</b></span>
                    <span>⬇ Downtime: <b style="color:#C8D8E8">{rec['downtime']}</b></span>
                    <span>👷 {rec['technician']}</span>
                    <span>✅ Confidence: <b style="color:#00E676">{rec['confidence']}</b></span>
                </div>
                <div style="font-size:0.7rem;color:#4A6585;margin-top:4px">
                    Parts: {', '.join(rec['parts'])}
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Trend chart ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.plotly_chart(health_trend_chart(trend), use_container_width=True)
