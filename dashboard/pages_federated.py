"""dashboard/pages_federated.py — Edge AI Nodes & Federated Learning pages."""
import streamlit as st
import time
from dashboard.data_sim import get_federated_clients, get_fl_training_history, get_edge_node_stats
from dashboard.styles import page_header, status_badge
from dashboard.charts import fl_convergence_chart


def render_edge_nodes():
    st.markdown(page_header("⚡", "Edge AI Nodes",
        "Local inference status — no raw data leaves the airline"), unsafe_allow_html=True)

    stats = get_edge_node_stats()
    clients = get_federated_clients()

    # ── Edge node KPIs ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(0,60,120,0.4),rgba(0,191,255,0.08));
        border:1px solid rgba(0,191,255,0.25);border-radius:14px;padding:16px 20px;margin-bottom:16px">
        <div style="display:flex;gap:20px;flex-wrap:wrap;align-items:center">
            <div style="font-size:0.8rem;font-weight:700;color:#00BFFF;text-transform:uppercase;
                letter-spacing:0.08em">🔒 EDGE DEPLOYMENT MODE</div>
            <span style="color:#00E676;font-size:0.78rem">✅ Running on Airline Edge Server</span>
            <span style="color:#00E676;font-size:0.78rem">✅ Cloud Independent</span>
            <span style="color:#00E676;font-size:0.78rem">✅ Offline Prediction Enabled</span>
            <span style="color:#00E676;font-size:0.78rem">✅ Instant Local Decision Making</span>
        </div>
    </div>""", unsafe_allow_html=True)

    cols = st.columns(4)
    metrics = [
        ("Local Inference Latency", f"{stats['latency_ms']} ms", "#00E676"),
        ("Predictions Today",       f"{stats['predictions_today']:,}", "#00BFFF"),
        ("CPU Usage",               f"{stats['cpu']}%", "#FFB300"),
        ("Memory Usage",            f"{stats['memory']}%", "#9B59B6"),
    ]
    for col, (label, val, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div style="background:rgba(0,20,40,0.7);border:1px solid rgba(0,191,255,0.15);
                border-radius:12px;padding:16px;text-align:center">
                <div style="font-size:0.7rem;color:#6B8CAE;text-transform:uppercase;letter-spacing:0.06em">{label}</div>
                <div style="font-size:1.8rem;font-weight:700;color:{color};margin:4px 0">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-airline edge nodes ───────────────────────────────────────────────
    st.markdown("#### 🛰 Airline Edge Node Status")
    node_cols = st.columns(5)
    for i, client in enumerate(clients):
        conn_color = {"Excellent":"#00E676","Good":"#00BFFF","Fair":"#FFB300"}.get(client["connection"],"#888")
        with node_cols[i]:
            st.markdown(f"""
            <div style="background:rgba(0,20,40,0.75);border:1px solid rgba(0,191,255,0.18);
                border-radius:12px;padding:14px 12px;text-align:center">
                <div style="font-size:1.8rem;margin-bottom:6px">🖥️</div>
                <div style="font-size:0.78rem;font-weight:700;color:#E8F4FF">{client['airline']}</div>
                <div style="font-size:0.65rem;color:#6B8CAE;margin:4px 0">Edge Server Node {i+1}</div>
                <div style="margin:6px 0">
                    <span style="color:#00E676;font-size:0.7rem">● {client['status']}</span>
                </div>
                <div style="font-size:0.68rem;color:#4A6585">Model: {client['model_version']}</div>
                <div style="font-size:0.68rem;color:#4A6585">Engines: {client['local_engines']}</div>
                <div style="font-size:0.68rem;margin-top:6px">
                    <span style="color:{conn_color}">Signal: {client['connection']}</span>
                </div>
                <div style="margin-top:8px;background:rgba(0,230,118,0.15);border:1px solid #00E67640;
                    border-radius:6px;padding:4px;font-size:0.65rem;color:#00E676">
                    🔒 Data: LOCAL ONLY
                </div>
                <div style="height:3px;background:rgba(0,191,255,0.15);border-radius:2px;margin-top:8px">
                    <div style="height:3px;background:#00BFFF;border-radius:2px;width:{60+i*8}%"></div>
                </div>
                <div style="font-size:0.6rem;color:#4A6585;margin-top:2px">{60+i*8}% utilization</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Inference flow diagram ───────────────────────────────────────────────
    st.markdown("#### 🔄 Edge AI Inference Pipeline")
    st.markdown("""
    <div style="background:rgba(0,20,40,0.6);border:1px solid rgba(0,191,255,0.15);
        border-radius:12px;padding:20px">
        <div style="display:flex;align-items:center;justify-content:center;gap:8px;flex-wrap:wrap">
            <div style="background:rgba(0,80,160,0.5);border:1px solid rgba(0,191,255,0.3);
                border-radius:8px;padding:8px 14px;text-align:center;font-size:0.75rem;color:#E8F4FF">
                ✈️<br><b>Aircraft Sensors</b>
            </div>
            <div style="color:#00BFFF;font-size:1.2rem">→</div>
            <div style="background:rgba(0,80,160,0.5);border:1px solid rgba(0,191,255,0.3);
                border-radius:8px;padding:8px 14px;text-align:center;font-size:0.75rem;color:#E8F4FF">
                🖥️<br><b>Edge Server</b><br><span style="color:#6B8CAE;font-size:0.65rem">On-site</span>
            </div>
            <div style="color:#00BFFF;font-size:1.2rem">→</div>
            <div style="background:rgba(0,60,120,0.6);border:1px solid #00BFFF;
                border-radius:8px;padding:8px 14px;text-align:center;font-size:0.75rem;color:#00BFFF">
                🧠<br><b>LSTM Inference</b><br><span style="color:#6B8CAE;font-size:0.65rem">Local &lt;5ms</span>
            </div>
            <div style="color:#00BFFF;font-size:1.2rem">→</div>
            <div style="background:rgba(0,80,160,0.5);border:1px solid rgba(0,191,255,0.3);
                border-radius:8px;padding:8px 14px;text-align:center;font-size:0.75rem;color:#E8F4FF">
                📊<br><b>RUL + Health</b><br><span style="color:#6B8CAE;font-size:0.65rem">Real-time</span>
            </div>
            <div style="color:#00BFFF;font-size:1.2rem">→</div>
            <div style="background:rgba(0,80,0,0.4);border:1px solid rgba(0,230,118,0.3);
                border-radius:8px;padding:8px 14px;text-align:center;font-size:0.75rem;color:#00E676">
                🔧<br><b>Maintenance Alert</b><br><span style="color:#6B8CAE;font-size:0.65rem">Auto-generated</span>
            </div>
        </div>
        <div style="text-align:center;margin-top:14px;font-size:0.72rem;color:#6B8CAE">
            ⛔ <b style="color:#FF4444">NO RAW DATA TRANSMITTED TO CLOUD</b>
            — Only local inference, privacy preserved
        </div>
    </div>""", unsafe_allow_html=True)


def render_federated_learning():
    st.markdown(page_header("🌐", "Federated Learning Hub",
        "Collaborative model training across airline fleets — privacy preserved"), unsafe_allow_html=True)

    clients = get_federated_clients()
    history = get_fl_training_history()

    # Privacy banner
    st.markdown("""
    <div style="background:rgba(0,60,0,0.3);border:1px solid rgba(0,230,118,0.3);
        border-radius:10px;padding:12px 18px;margin-bottom:16px;display:flex;gap:30px;flex-wrap:wrap">
        <div>
            <span style="color:#00E676;font-weight:700;font-size:0.85rem">🔒 Privacy Guarantee</span>
        </div>
        <div style="font-size:0.8rem">
            <span style="color:#6B8CAE">Raw Sensor Data Shared: </span>
            <span style="color:#FF4444;font-weight:700">NO — NEVER</span>
        </div>
        <div style="font-size:0.8rem">
            <span style="color:#6B8CAE">Model Parameters Shared: </span>
            <span style="color:#00E676;font-weight:700">YES — Encrypted Weights Only</span>
        </div>
        <div style="font-size:0.8rem">
            <span style="color:#6B8CAE">Aggregation: </span>
            <span style="color:#00BFFF;font-weight:700">FedAvgRUL Strategy</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # FL convergence chart
    st.plotly_chart(fl_convergence_chart(history), use_container_width=True)

    # ── Client cards ─────────────────────────────────────────────────────────
    st.markdown("#### 🏢 Airline Federated Clients")
    client_cols = st.columns(5)
    for i, client in enumerate(clients):
        train_color = {"Complete":"#00E676","Training":"#FFB300","Idle":"#6B8CAE"}.get(client['local_training'],'#888')
        with client_cols[i]:
            st.markdown(f"""
            <div style="background:rgba(0,20,40,0.8);border:1px solid rgba(0,191,255,0.18);
                border-radius:12px;padding:14px 10px">
                <div style="text-align:center;font-size:1.5rem;margin-bottom:6px">🏢</div>
                <div style="font-size:0.75rem;font-weight:700;color:#E8F4FF;text-align:center">{client['airline']}</div>
                <div style="height:1px;background:rgba(0,191,255,0.1);margin:8px 0"></div>
                <div style="font-size:0.68rem;display:flex;flex-direction:column;gap:4px">
                    <div><span style="color:#6B8CAE">Status:</span>
                        <span style="color:#00E676;margin-left:4px">● {client['status']}</span></div>
                    <div><span style="color:#6B8CAE">Training:</span>
                        <span style="color:{train_color};margin-left:4px">{client['local_training']}</span></div>
                    <div><span style="color:#6B8CAE">Version:</span>
                        <span style="color:#E8F4FF;margin-left:4px">{client['model_version']}</span></div>
                    <div><span style="color:#6B8CAE">Sync:</span>
                        <span style="color:#E8F4FF;margin-left:4px">{client['last_sync']}</span></div>
                    <div><span style="color:#6B8CAE">Engines:</span>
                        <span style="color:#00BFFF;margin-left:4px">{client['local_engines']}</span></div>
                    <div><span style="color:#6B8CAE">Privacy:</span>
                        <span style="color:#00E676;margin-left:4px">🔒 {client['privacy']}</span></div>
                </div>
                <div style="margin-top:8px;text-align:center">
                    <div style="background:#FF444415;border:1px solid #FF444440;border-radius:4px;
                        padding:3px;font-size:0.63rem;color:#FF4444">Raw Data: NOT SHARED</div>
                    <div style="background:#00E67615;border:1px solid #00E67640;border-radius:4px;
                        padding:3px;font-size:0.63rem;color:#00E676;margin-top:3px">Model Weights: ✓ SHARED</div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── FL Workflow diagram ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔄 Federated Learning Workflow")
    steps = [
        ("✈️", "Aircraft Sensors"),
        ("🖥️", "Airline Edge Server"),
        ("🔄", "Local Training"),
        ("📦", "Model Weights"),
        ("⚡", "Global Aggregation"),
        ("🧠", "Updated Global Model"),
        ("📡", "Edge Deployment"),
    ]
    step_cols = st.columns(len(steps))
    for i, (col, (icon, label)) in enumerate(zip(step_cols, steps)):
        with col:
            connector = "→" if i < len(steps)-1 else ""
            st.markdown(f"""
            <div style="text-align:center">
                <div style="background:rgba(0,40,80,0.7);border:1px solid rgba(0,191,255,0.25);
                    border-radius:10px;padding:10px 6px;font-size:0.72rem;color:#E8F4FF">
                    <div style="font-size:1.4rem;margin-bottom:4px">{icon}</div>
                    {label}
                </div>
            </div>""", unsafe_allow_html=True)
