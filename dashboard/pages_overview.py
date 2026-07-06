"""dashboard/pages_overview.py — Dashboard landing & Fleet Health pages."""
import streamlit as st
import pandas as pd
import time
from dashboard.data_sim import make_fleet, get_alerts
from dashboard.styles import page_header, kpi_card, risk_badge, status_badge
from dashboard.charts import fleet_risk_donut


def render_dashboard(df_f, df_c):
    st.markdown(page_header("📡", "Mission Control Dashboard",
        "Real-time Aircraft Fleet Health Monitoring — Edge AI & Federated Learning"), unsafe_allow_html=True)

    fleet = make_fleet(seed_offset=int(time.time() // 30))
    alerts = get_alerts()

    # ── Top KPI row ────────────────────────────────────────────────────────────
    cols = st.columns(8)
    kpis = [
        ("Fleet Health Score", "84.2%",  "↓ 1.3 from yesterday", "#00E676", "🏥"),
        ("Aircraft Active",    "10",      "All units reporting",   "#00BFFF", "✈️"),
        ("Need Inspection",    "3",       "AX-102, BX-214, CX-319","#FF4444", "⚠️"),
        ("FL Clients",         "5",       "All airlines online",   "#00BFFF", "🌐"),
        ("Model Version",      "v2.7.3",  "Updated 14 min ago",    "#9B59B6", "🧠"),
        ("Last FL Update",     "14 min",  "Round 20 complete",     "#FFB300", "🔄"),
        ("Edge Nodes Online",  "10 / 10", "100% availability",     "#00E676", "⚡"),
        ("Avg RUL",            "67 cyc",  "Fleet average",         "#00BFFF", "⏱️"),
    ]
    for col, (label, val, sub, color, icon) in zip(cols, kpis):
        with col:
            st.markdown(kpi_card(label, val, sub, color, icon), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Fleet table + donut ────────────────────────────────────────────────────
    left, right = st.columns([3, 1])

    with left:
        st.markdown("#### 📋 Live Aircraft Fleet Status")

        # Color-coded dataframe
        df_fleet = pd.DataFrame(fleet)
        risk_color = {"Critical":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"}
        df_fleet["Risk"] = df_fleet["Risk"].map(lambda r: f"{risk_color.get(r,'')} {r}")
        df_fleet["Edge AI"] = df_fleet["Edge AI"].map(lambda s: "✅ Active" if s=="Active" else "❌ Offline")
        df_fleet["Maintenance"] = df_fleet["Maintenance"].map(
            lambda s: "🔧 Scheduled" if s=="Scheduled" else "✅ OK")

        st.dataframe(df_fleet, use_container_width=True, height=360,
                     column_config={
                         "Health %": st.column_config.ProgressColumn("Health %", min_value=0, max_value=100),
                         "RUL (cycles)": st.column_config.NumberColumn("RUL", format="%d cyc"),
                     })

    with right:
        st.plotly_chart(fleet_risk_donut(make_fleet()), use_container_width=True)

    # ── Recent Alerts ──────────────────────────────────────────────────────────
    st.markdown("#### 🔔 Recent Alerts")
    for a in alerts[:4]:
        lvl_colors = {"Critical":"#FF0055","High":"#FF4444","Medium":"#FFB300","Low":"#00E676"}
        c = lvl_colors.get(a["level"], "#888")
        st.markdown(f"""
        <div style="background:rgba(0,20,40,0.6);border-left:3px solid {c};
            border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:6px;
            display:flex;align-items:center;gap:12px;">
            <span style="font-size:1.1rem">{a['icon']}</span>
            <div style="flex:1">
                <span style="color:{c};font-weight:600;font-size:0.82rem">[{a['id']}]</span>
                <span style="color:#C8D8E8;font-size:0.82rem;margin-left:6px">{a['msg']}</span>
            </div>
            <span style="color:#4A6585;font-size:0.72rem">{a['time']}</span>
        </div>""", unsafe_allow_html=True)


def render_fleet_health(df_f, df_c):
    st.markdown(page_header("🛫", "Fleet Health Overview",
        "Comprehensive health status across all monitored aircraft"), unsafe_allow_html=True)

    fleet = make_fleet(seed_offset=int(time.time() // 60))

    # Health distribution bars
    st.markdown("#### Health Score Distribution by Aircraft")
    cols = st.columns(5)
    for i, ac in enumerate(fleet[:5]):
        with cols[i % 5]:
            score = ac["Health %"]
            color = "#00E676" if score > 70 else ("#FFB300" if score > 40 else "#FF4444")
            st.markdown(f"""
            <div style="background:rgba(0,20,40,0.7);border:1px solid rgba(0,191,255,0.15);
                border-radius:10px;padding:14px 10px;text-align:center;margin-bottom:10px">
                <div style="font-size:0.75rem;color:#6B8CAE;font-weight:600">{ac['Aircraft ID']}</div>
                <div style="font-size:0.65rem;color:#4A6585;margin-bottom:6px">{ac['Type']}</div>
                <div style="font-size:1.5rem;font-weight:700;color:{color}">{score}%</div>
                <div style="font-size:0.65rem;color:#4A6585">RUL: {ac['RUL (cycles)']} cyc</div>
                <div style="height:4px;background:rgba(255,255,255,0.1);border-radius:2px;margin-top:6px">
                  <div style="height:4px;background:{color};border-radius:2px;width:{score}%"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    cols2 = st.columns(5)
    for i, ac in enumerate(fleet[5:]):
        with cols2[i % 5]:
            score = ac["Health %"]
            color = "#00E676" if score > 70 else ("#FFB300" if score > 40 else "#FF4444")
            st.markdown(f"""
            <div style="background:rgba(0,20,40,0.7);border:1px solid rgba(0,191,255,0.15);
                border-radius:10px;padding:14px 10px;text-align:center;margin-bottom:10px">
                <div style="font-size:0.75rem;color:#6B8CAE;font-weight:600">{ac['Aircraft ID']}</div>
                <div style="font-size:0.65rem;color:#4A6585;margin-bottom:6px">{ac['Type']}</div>
                <div style="font-size:1.5rem;font-weight:700;color:{color}">{score}%</div>
                <div style="font-size:0.65rem;color:#4A6585">RUL: {ac['RUL (cycles)']} cyc</div>
                <div style="height:4px;background:rgba(255,255,255,0.1);border-radius:2px;margin-top:6px">
                  <div style="height:4px;background:{color};border-radius:2px;width:{score}%"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Full Fleet Data Table")
    df_fleet = pd.DataFrame(fleet)
    st.dataframe(df_fleet, use_container_width=True, height=400,
                 column_config={
                     "Health %": st.column_config.ProgressColumn("Health %", min_value=0, max_value=100),
                     "RUL (cycles)": st.column_config.NumberColumn("RUL", format="%d cyc"),
                 })
