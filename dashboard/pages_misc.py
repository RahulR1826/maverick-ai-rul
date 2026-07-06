"""dashboard/pages_misc.py — Alerts, Analytics, Maintenance, Settings pages."""
import streamlit as st
import pandas as pd
import numpy as np
from dashboard.data_sim import (get_alerts, get_fl_training_history,
                                 get_maintenance_recommendations, AIRCRAFT_IDS)
from dashboard.styles import page_header
from dashboard.charts import fl_convergence_chart, prediction_scatter


def render_maintenance():
    st.markdown(page_header("🔧", "Maintenance Center",
        "AI-generated maintenance schedules and recommendations"), unsafe_allow_html=True)

    selected = st.selectbox("Select Aircraft for Maintenance Review", AIRCRAFT_IDS, key="maint_select")
    recs = get_maintenance_recommendations(selected)

    priority_colors = {"URGENT":"#FF0055","HIGH":"#FF4444","MEDIUM":"#FFB300","LOW":"#00E676"}

    for rec in recs:
        color = priority_colors.get(rec["priority"], "#888")
        with st.container():
            st.markdown(f"""
            <div style="background:rgba(0,15,30,0.8);border:1px solid rgba(0,191,255,0.15);
                border-left:4px solid {color};border-radius:0 12px 12px 0;padding:16px 20px;margin-bottom:12px">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
                    <div>
                        <span style="color:#E8F4FF;font-weight:700;font-size:0.92rem">{rec['action']}</span>
                        <span style="margin-left:10px;background:{color}25;color:{color};
                            padding:2px 10px;border-radius:20px;font-size:0.7rem;font-weight:700">{rec['priority']}</span>
                    </div>
                    <span style="color:#00E676;font-weight:700;font-size:0.85rem">Confidence: {rec['confidence']}</span>
                </div>
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px">
                    <div style="background:rgba(0,30,60,0.5);border-radius:8px;padding:10px">
                        <div style="font-size:0.65rem;color:#6B8CAE;margin-bottom:4px">⏱ WINDOW</div>
                        <div style="font-size:0.82rem;color:#E8F4FF;font-weight:600">{rec['window']}</div>
                    </div>
                    <div style="background:rgba(0,30,60,0.5);border-radius:8px;padding:10px">
                        <div style="font-size:0.65rem;color:#6B8CAE;margin-bottom:4px">⬇ DOWNTIME</div>
                        <div style="font-size:0.82rem;color:#E8F4FF;font-weight:600">{rec['downtime']}</div>
                    </div>
                    <div style="background:rgba(0,30,60,0.5);border-radius:8px;padding:10px">
                        <div style="font-size:0.65rem;color:#6B8CAE;margin-bottom:4px">👷 TECHNICIAN</div>
                        <div style="font-size:0.82rem;color:#E8F4FF;font-weight:600">{rec['technician']}</div>
                    </div>
                    <div style="background:rgba(0,30,60,0.5);border-radius:8px;padding:10px">
                        <div style="font-size:0.65rem;color:#6B8CAE;margin-bottom:4px">🔩 PARTS</div>
                        <div style="font-size:0.78rem;color:#E8F4FF">{', '.join(rec['parts'])}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📅 Maintenance Schedule Overview")
    schedule_data = {
        "Aircraft": AIRCRAFT_IDS,
        "Next Maintenance": ["2 days","5 days","12 days","7 days","1 day","18 days","9 days","22 days","3 days","15 days"],
        "Type": ["HPC Inspect","Oil Change","Borescope","Fan Blade","URGENT Inspect","Routine","Seal Check","Routine","Vibration","Routine"],
        "Priority": ["HIGH","LOW","MEDIUM","MEDIUM","CRITICAL","LOW","MEDIUM","LOW","HIGH","LOW"],
        "Technician": ["L3","L1","L3","L2","L3","L1","L2","L1","L2","L1"],
    }
    st.dataframe(pd.DataFrame(schedule_data), use_container_width=True)


def render_alerts():
    st.markdown(page_header("🔔", "Alert Center",
        "Real-time anomaly detection and maintenance alerts"), unsafe_allow_html=True)

    alerts = get_alerts()
    lvl_filter = st.multiselect("Filter by Level", ["Critical","High","Medium","Low"],
                                 default=["Critical","High","Medium","Low"])

    filtered = [a for a in alerts if a["level"] in lvl_filter]
    lvl_colors = {"Critical":"#FF0055","High":"#FF4444","Medium":"#FFB300","Low":"#00E676"}

    for a in filtered:
        c = lvl_colors.get(a["level"], "#888")
        st.markdown(f"""
        <div style="background:rgba(0,15,30,0.8);border:1px solid rgba(0,191,255,0.12);
            border-left:4px solid {c};border-radius:0 12px 12px 0;
            padding:14px 18px;margin-bottom:8px">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div style="display:flex;align-items:center;gap:10px">
                    <span style="font-size:1.3rem">{a['icon']}</span>
                    <div>
                        <div>
                            <span style="color:{c};font-weight:700;font-size:0.8rem">{a['level'].upper()}</span>
                            <span style="color:#6B8CAE;font-size:0.75rem;margin:0 8px">|</span>
                            <span style="color:#00BFFF;font-weight:600;font-size:0.82rem">{a['id']}</span>
                        </div>
                        <div style="color:#C8D8E8;font-size:0.82rem;margin-top:3px">{a['msg']}</div>
                    </div>
                </div>
                <div style="text-align:right">
                    <div style="color:#4A6585;font-size:0.72rem">{a['time']}</div>
                    <div style="margin-top:4px">
                        <span style="background:{c}20;color:{c};padding:2px 8px;
                            border-radius:4px;font-size:0.68rem">{a['level']}</span>
                    </div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)


def render_analytics(df_f, df_c):
    st.markdown(page_header("📊", "Analytics",
        "Fleet-wide performance metrics and model accuracy analysis"), unsafe_allow_html=True)

    if df_f is not None and df_c is not None:
        # Accuracy metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        from sklearn.metrics import r2_score
        rmse_f = np.sqrt(mean_squared_error(df_f["y_true"], df_f["y_pred"]))
        mae_f  = mean_absolute_error(df_f["y_true"], df_f["y_pred"])
        r2_f   = r2_score(df_f["y_true"], df_f["y_pred"])
        rmse_c = np.sqrt(mean_squared_error(df_c["y_true"], df_c["y_pred"]))
        r2_c   = r2_score(df_c["y_true"], df_c["y_pred"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Federated RMSE", f"{rmse_f:.2f} cyc", f"{rmse_c - rmse_f:+.2f} vs centralized")
        col2.metric("Federated MAE",  f"{mae_f:.2f} cyc")
        col3.metric("R² Accuracy",    f"{r2_f*100:.1f}%",  f"+{(r2_f-r2_c)*100:.1f}% vs centralized")
        col4.metric("Model",          "v2.7.3",             "Round 20 complete")

        st.markdown("---")

        left, right = st.columns(2)
        with left:
            st.plotly_chart(prediction_scatter(df_f, df_c), use_container_width=True)
        with right:
            hist = get_fl_training_history()
            st.plotly_chart(fl_convergence_chart(hist), use_container_width=True)

        # Error distribution
        st.markdown("#### Error Distribution")
        import plotly.graph_objects as go
        errors_f = df_f["y_pred"] - df_f["y_true"]
        errors_c = df_c["y_pred"] - df_c["y_true"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=errors_f, name="Federated", opacity=0.75,
                                    marker_color="#00BFFF", nbinsx=20))
        fig.add_trace(go.Histogram(x=errors_c, name="Centralized", opacity=0.65,
                                    marker_color="#FFB300", nbinsx=20))
        fig.update_layout(
            title=dict(text="Prediction Error Distribution", font=dict(color="#E8F4FF", size=13)),
            xaxis_title="Error (cycles)", yaxis_title="Count", barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9BB8D4"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=40, b=10), height=300,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_settings():
    st.markdown(page_header("⚙️", "Settings",
        "Configure Edge AI, Federated Learning, and system parameters"), unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌐 Federated Learning Settings")
        st.number_input("FL Communication Rounds", min_value=5, max_value=100, value=20)
        st.number_input("Local Epochs per Round", min_value=1, max_value=20, value=5)
        st.number_input("Number of Clients", min_value=2, max_value=20, value=5)
        st.slider("Fraction Fit (clients per round)", 0.1, 1.0, 0.8)
        st.toggle("Non-IID Data Distribution", value=False)
        st.number_input("Dirichlet Alpha (Non-IID)", min_value=0.1, max_value=2.0, value=0.5)

    with col2:
        st.markdown("#### 🧠 Model & Prediction Settings")
        st.text_input("Model Version", value="v2.7.3")
        st.text_input("Edge Deployment Version", value="edge-v2.7.3")
        st.slider("RUL Prediction Threshold (cycles)", 5, 50, 25,
                   help="Alert when RUL drops below this")
        st.slider("Maintenance Alert Threshold (cycles)", 5, 40, 20)
        st.toggle("Explainability (XAI) Enabled", value=True)
        st.toggle("Differential Privacy", value=False)

    st.markdown("---")
    st.markdown("#### 🖥️ Edge Node Configuration")
    col3, col4 = st.columns(2)
    with col3:
        st.toggle("Offline Mode (Edge-Only Inference)", value=True)
        st.number_input("Inference Batch Size", value=32)
        st.selectbox("Device Target", ["CPU (Edge Server)", "GPU (Edge Server)", "ONNX Runtime"])
    with col4:
        st.toggle("Auto-sync after FL Round", value=True)
        st.number_input("Sync Interval (minutes)", value=30)
        st.text_input("Global Aggregation Server", value="fl-server.maverick.internal")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("💾 Save Configuration", use_container_width=True):
        st.success("✅ Configuration saved and pushed to all Edge Nodes")
