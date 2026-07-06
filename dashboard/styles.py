"""
dashboard/styles.py
--------------------
All CSS and HTML theming for Maverick AI — aerospace dark theme.
"""

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary:   #070B14;
    --bg-secondary: #0D1526;
    --bg-card:      rgba(13, 21, 38, 0.85);
    --bg-glass:     rgba(20, 35, 65, 0.6);
    --accent-blue:  #00BFFF;
    --accent-cyan:  #00FFFF;
    --accent-glow:  rgba(0, 191, 255, 0.15);
    --text-primary: #E8F4FF;
    --text-muted:   #6B8CAE;
    --border:       rgba(0, 191, 255, 0.18);
    --success:      #00E676;
    --warning:      #FFB300;
    --danger:       #FF4444;
    --critical:     #FF0055;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* Let Streamlit handle its own header and sidebar toggle to prevent accessibility issues */
footer { visibility: hidden; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--accent-blue); border-radius: 4px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070B14 0%, #0A1628 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: var(--text-primary) !important;
    font-size: 0.87rem !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: var(--text-muted) !important;
}

/* Main content */
.main .block-container {
    background: var(--bg-primary) !important;
    padding: 1.5rem 2rem !important;
    max-width: 1600px;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    backdrop-filter: blur(12px) !important;
}
[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent-blue) !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* DataFrames / Tables */
[data-testid="stDataFrame"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Plotly charts */
.js-plotly-plot .plotly .bg {
    fill: transparent !important;
}

/* Dividers */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.2rem 0 !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #003366, #005599) !important;
    border: 1px solid var(--accent-blue) !important;
    border-radius: 8px !important;
    color: var(--accent-blue) !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    background: linear-gradient(135deg, #005599, #0077CC) !important;
    box-shadow: 0 0 16px var(--accent-glow) !important;
    transform: translateY(-1px) !important;
}

/* Progress bars */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
    border-radius: 4px !important;
}

/* Headers */
h1, h2, h3 { color: var(--text-primary) !important; }
</style>
"""

def page_header(icon, title, subtitle=""):
    return f"""
    <div style="
        display:flex; align-items:center; gap:14px;
        padding:1.2rem 1.5rem;
        background: linear-gradient(135deg, rgba(0,80,160,0.3), rgba(0,191,255,0.08));
        border: 1px solid rgba(0,191,255,0.2);
        border-radius: 14px;
        margin-bottom: 1.5rem;
    ">
        <span style="font-size:2.2rem">{icon}</span>
        <div>
            <div style="font-size:1.5rem; font-weight:700; color:#E8F4FF; letter-spacing:-0.02em">{title}</div>
            <div style="font-size:0.82rem; color:#6B8CAE; margin-top:2px">{subtitle}</div>
        </div>
        <div style="margin-left:auto; text-align:right;">
            <div style="font-size:0.7rem; color:#6B8CAE; text-transform:uppercase; letter-spacing:0.08em">MAVERICK AI</div>
            <div style="font-size:0.65rem; color:#00BFFF; opacity:0.8">Edge AI · Federated Learning · XAI</div>
        </div>
    </div>
    """

def kpi_card(label, value, sub="", color="#00BFFF", icon=""):
    return f"""
    <div style="
        background: rgba(0,30,60,0.7);
        border: 1px solid rgba(0,191,255,0.2);
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        backdrop-filter: blur(10px);
        position: relative; overflow: hidden;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    ">
        <div style="position:absolute;top:0;right:0;width:80px;height:80px;
            background:radial-gradient(circle at top right, {color}18, transparent 70%);
            border-radius:0 12px 0 100%"></div>
        <div style="font-size:1.4rem; margin-bottom:4px">{icon}</div>
        <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em; color:#6B8CAE">{label}</div>
        <div style="font-size:1.7rem; font-weight:700; color:{color}; margin:4px 0; line-height:1">{value}</div>
        <div style="font-size:0.72rem; color:#4A6585">{sub}</div>
    </div>
    """

def risk_badge(risk):
    colors = {
        "Critical": ("#FF0055", "#FF005520"),
        "High":     ("#FF4444", "#FF444420"),
        "Medium":   ("#FFB300", "#FFB30020"),
        "Low":      ("#00E676", "#00E67620"),
    }
    fg, bg = colors.get(risk, ("#888", "#88888820"))
    return f'<span style="color:{fg};background:{bg};padding:2px 8px;border-radius:20px;font-size:0.72rem;font-weight:600;border:1px solid {fg}50">{risk}</span>'

def status_badge(status, active_label="Active"):
    if status in ["Active", "Online", "Complete"]:
        return f'<span style="color:#00E676;background:#00E67618;padding:2px 8px;border-radius:20px;font-size:0.72rem;font-weight:600;border:1px solid #00E67640">● {status}</span>'
    elif status in ["Offline", "Error"]:
        return f'<span style="color:#FF4444;background:#FF444418;padding:2px 8px;border-radius:20px;font-size:0.72rem;font-weight:600;border:1px solid #FF444440">● {status}</span>'
    else:
        return f'<span style="color:#FFB300;background:#FFB30018;padding:2px 8px;border-radius:20px;font-size:0.72rem;font-weight:600;border:1px solid #FFB30040">● {status}</span>'

SIDEBAR_LOGO = """
<div style="text-align:center; padding: 1.2rem 0.5rem 0.5rem;">
    <div style="font-size:1.6rem; font-weight:800; letter-spacing:-0.03em;
        background: linear-gradient(135deg, #00BFFF, #00FFFF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        ✈ MAVERICK AI
    </div>
    <div style="font-size:0.67rem; color:#4A6585; text-transform:uppercase; letter-spacing:0.1em; margin-top:4px">
        Edge AI · Predictive Maintenance
    </div>
    <div style="height:1px; background: linear-gradient(90deg, transparent, #00BFFF40, transparent); margin:10px 0;"></div>
</div>
"""

FOOTER_HTML = """
<div style="
    position: fixed; bottom: 0; left: 0; right: 0;
    background: rgba(7,11,20,0.95);
    border-top: 1px solid rgba(0,191,255,0.15);
    padding: 6px 24px;
    display: flex; align-items: center; gap: 24px;
    font-size: 0.68rem; color: #4A6585; z-index: 999;
    backdrop-filter: blur(10px);
">
    <span style="color:#00E676">⬤ Edge AI Enabled</span>
    <span style="color:#00E676">⬤ Federated Learning Connected</span>
    <span style="color:#00BFFF">🔒 Data Privacy Protected</span>
    <span>Last Global Model Update: 14 min ago</span>
    <span>Model: v2.7.3</span>
    <span style="margin-left:auto; color:#00E676">⬤ System Operational</span>
</div>
"""
