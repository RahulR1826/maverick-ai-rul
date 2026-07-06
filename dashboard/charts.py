"""
dashboard/charts.py
--------------------
All Plotly chart builders for Maverick AI dashboard.
Returns go.Figure objects with the aerospace dark theme applied.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def apply_dark_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#9BB8D4", size=11),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    fig.update_xaxes(gridcolor="rgba(0,191,255,0.08)", zeroline=False, color="#6B8CAE")
    fig.update_yaxes(gridcolor="rgba(0,191,255,0.08)", zeroline=False, color="#6B8CAE")
    return fig


def rul_gauge(rul_value, max_rul=130):
    if rul_value > 60: color, label = "#00E676", "HEALTHY"
    elif rul_value > 25: color, label = "#FFB300", "WATCH"
    else: color, label = "#FF4444", "CRITICAL"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=rul_value,
        delta={"reference": max_rul, "valueformat": ".0f", "suffix": " from max"},
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"<b>REMAINING USEFUL LIFE</b><br><span style='font-size:0.8em;color:{color}'>{label}</span>",
               "font": {"color": "#E8F4FF", "size": 14}},
        number={"suffix": " cycles", "font": {"color": color, "size": 40}},
        gauge={
            "axis": {"range": [0, max_rul], "tickcolor": "#6B8CAE", "tickwidth": 1,
                     "tickfont": {"color": "#6B8CAE", "size": 9}},
            "bar": {"color": color, "thickness": 0.22},
            "bgcolor": "rgba(0,20,40,0.5)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "rgba(255,68,68,0.18)"},
                {"range": [25, 60], "color": "rgba(255,179,0,0.15)"},
                {"range": [60, max_rul], "color": "rgba(0,230,118,0.12)"},
            ],
            "threshold": {"line": {"color": "rgba(255,255,255,0.5)", "width": 2},
                          "thickness": 0.85, "value": 25},
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#9BB8D4",
                      margin=dict(l=20, r=20, t=50, b=20), height=280)
    return fig


def health_gauge(score):
    if score > 70: color = "#00E676"
    elif score > 40: color = "#FFB300"
    else: color = "#FF4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "<b>HEALTH SCORE</b>", "font": {"color": "#E8F4FF", "size": 13}},
        number={"suffix": "%", "font": {"color": color, "size": 38}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#6B8CAE", "tickfont": {"size": 9}},
            "bar": {"color": color, "thickness": 0.22},
            "bgcolor": "rgba(0,20,40,0.5)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(255,68,68,0.15)"},
                {"range": [40, 70], "color": "rgba(255,179,0,0.12)"},
                {"range": [70, 100], "color": "rgba(0,230,118,0.1)"},
            ],
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#9BB8D4",
                      margin=dict(l=20, r=20, t=45, b=20), height=240)
    return fig


def feature_importance_chart(data):
    sensors = [d["sensor"] for d in data]
    scores = [d["importance"] for d in data]
    colors = [f"rgba(0,{int(191 * s/max(scores))},255,0.85)" for s in scores]

    fig = go.Figure(go.Bar(
        x=scores, y=sensors, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside", textfont=dict(color="#9BB8D4", size=10),
    ))
    fig.update_layout(
        title=dict(text="Feature Importance (XAI)", font=dict(color="#E8F4FF", size=13)),
        xaxis=dict(title="Contribution (%)", range=[0, max(scores) * 1.25]),
        yaxis=dict(autorange="reversed"),
        height=320, margin=dict(l=160, r=60, t=40, b=10)
    )
    return apply_dark_theme(fig)


def fl_convergence_chart(history):
    rounds = history["rounds"]
    loss = history["loss"]
    rmse = history["rmse"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds, y=loss, name="Aggregated Loss",
        line=dict(color="#00BFFF", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,191,255,0.06)",
    ))
    fig.add_trace(go.Scatter(
        x=rounds, y=[r * 20 for r in rmse], name="Val RMSE × 20",
        line=dict(color="#00E676", width=2, dash="dot"),
        yaxis="y2"
    ))
    fig.update_layout(
        title=dict(text="Federated Learning Convergence", font=dict(color="#E8F4FF", size=13)),
        xaxis=dict(title="Round"),
        yaxis=dict(title="Loss"),
        yaxis2=dict(title="RMSE×20", overlaying="y", side="right", showgrid=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        height=300
    )
    return apply_dark_theme(fig)


def prediction_scatter(df_f, df_c):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_f["y_true"], y=df_f["y_pred"],
        mode="markers", name="Federated Model",
        marker=dict(color="#00BFFF", size=7, opacity=0.75,
                    line=dict(color="rgba(0,191,255,0.3)", width=1)),
    ))
    fig.add_trace(go.Scatter(
        x=df_c["y_true"], y=df_c["y_pred"],
        mode="markers", name="Centralized Model",
        marker=dict(color="#FFB300", size=7, opacity=0.65,
                    symbol="diamond"),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 130], y=[0, 130], mode="lines",
        name="Perfect Prediction",
        line=dict(color="rgba(255,255,255,0.35)", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        title=dict(text="Prediction Accuracy — True vs Predicted RUL", font=dict(color="#E8F4FF", size=13)),
        xaxis=dict(title="True RUL (cycles)"),
        yaxis=dict(title="Predicted RUL (cycles)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        height=380
    )
    return apply_dark_theme(fig)


def health_trend_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["days_ago"], y=data["health"],
        name="Health Score", fill="tozeroy",
        line=dict(color="#00E676", width=2),
        fillcolor="rgba(0,230,118,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=data["days_ago"], y=[r for r in data["rul"]],
        name="RUL (cycles)", line=dict(color="#00BFFF", width=2, dash="dot"),
        yaxis="y2",
    ))
    fig.update_layout(
        title=dict(text="30-Day Health & RUL Trend", font=dict(color="#E8F4FF", size=13)),
        xaxis=dict(title="Days Ago", autorange="reversed"),
        yaxis=dict(title="Health Score"),
        yaxis2=dict(title="RUL", overlaying="y", side="right", showgrid=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        height=300,
    )
    return apply_dark_theme(fig)


def fleet_risk_donut(fleet):
    from collections import Counter
    counts = Counter(a["Risk"] for a in fleet)
    labels = list(counts.keys())
    values = list(counts.values())
    color_map = {"Critical": "#FF0055", "High": "#FF4444", "Medium": "#FFB300", "Low": "#00E676"}
    colors = [color_map.get(l, "#888") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        textfont=dict(size=10, color="#E8F4FF"),
        hoverinfo="label+value+percent",
    ))
    fig.add_annotation(text=f"<b>{len(fleet)}</b><br><span style='font-size:9px'>Aircraft</span>",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color="#E8F4FF", size=18))
    fig.update_layout(
        title=dict(text="Fleet Risk Distribution", font=dict(color="#E8F4FF", size=13)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10), height=280,
    )
    return fig


def sensor_sparkline(values_history, label, color="#00BFFF"):
    fig = go.Figure(go.Scatter(
        x=list(range(len(values_history))), y=values_history,
        mode="lines", line=dict(color=color, width=1.5),
        fill="tozeroy", fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.08,)}",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), height=50,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig
