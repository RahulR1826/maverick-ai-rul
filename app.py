import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# Set page config for a wider layout
st.set_page_config(
    page_title="Aircraft Engine Health Monitoring",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
    }
    h1 {
        color: #4A90E2;
    }
    h2 {
        color: #F39C12;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    centralized_path = "results/centralized_FD001_predictions.csv"
    federated_path = "results/federated_FD001_predictions.csv"
    
    if not os.path.exists(centralized_path) or not os.path.exists(federated_path):
        st.error(f"Prediction files not found in the 'results/' directory. Please ensure you have run the evaluation.")
        return None, None
        
    df_c = pd.read_csv(centralized_path)
    df_f = pd.read_csv(federated_path)
    
    # Add Engine ID (Test engines are in order)
    df_c['Engine_ID'] = df_c.index + 1
    df_f['Engine_ID'] = df_f.index + 1
    
    return df_c, df_f

def calculate_metrics(df):
    rmse = np.sqrt(mean_squared_error(df['y_true'], df['y_pred']))
    mae = mean_absolute_error(df['y_true'], df['y_pred'])
    
    # NASA Score (Asymmetric scoring function)
    # Late predictions (overestimating RUL) are penalized more than early predictions
    d = df['y_pred'] - df['y_true']
    score = np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))
    
    return rmse, mae, score

# ---------------------------------------------------------------------------
# App Layout
# ---------------------------------------------------------------------------
st.title("✈️ Edge AI for Predictive Maintenance & Aircraft Health Monitoring")
st.markdown("Visualizing the Remaining Useful Life (RUL) of aircraft turbofan engines using Federated Learning (LSTM).")

df_c, df_f = load_data()

if df_c is not None and df_f is not None:
    
    # --- Sidebar ---
    st.sidebar.header("Navigation")
    view_mode = st.sidebar.radio("Select View:", 
                                 ["Overview & Metrics", "Engine-Specific Analysis", "Model Comparison Scatter"])
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Dataset:** NASA C-MAPSS (FD001)\n\n"
        "**Model:** Stacked LSTM\n\n"
        "**Federated Strategy:** FedAvgRUL"
    )

    # --- Calculations ---
    rmse_c, mae_c, score_c = calculate_metrics(df_c)
    rmse_f, mae_f, score_f = calculate_metrics(df_f)
    
    # --- 1. Overview Page ---
    if view_mode == "Overview & Metrics":
        st.header("📊 Fleet Health Overview")
        
        # Display Key Metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌐 Federated Model (Edge AI)")
            subcol1, subcol2, subcol3 = st.columns(3)
            subcol1.metric("RMSE (cycles)", f"{rmse_f:.2f}")
            subcol2.metric("MAE (cycles)", f"{mae_f:.2f}")
            subcol3.metric("NASA Score", f"{score_f:,.0f}")
            
        with col2:
            st.subheader("🏢 Centralized Model (Baseline)")
            subcol4, subcol5, subcol6 = st.columns(3)
            subcol4.metric("RMSE (cycles)", f"{rmse_c:.2f}")
            subcol5.metric("MAE (cycles)", f"{mae_c:.2f}")
            subcol6.metric("NASA Score", f"{score_c:,.0f}")
            
        st.markdown("---")
        
        # Comparison Bar Chart for the first 20 Engines
        st.subheader("Predicted RUL vs True RUL (First 20 Engines)")
        
        merged_df = pd.DataFrame({
            'Engine_ID': df_f['Engine_ID'][:20],
            'True_RUL': df_f['y_true'][:20],
            'Federated_Pred': df_f['y_pred'][:20],
            'Centralized_Pred': df_c['y_pred'][:20]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=merged_df['Engine_ID'], y=merged_df['True_RUL'], name='True RUL', marker_color='#4A90E2'))
        fig.add_trace(go.Bar(x=merged_df['Engine_ID'], y=merged_df['Federated_Pred'], name='Federated Pred', marker_color='#F39C12'))
        
        fig.update_layout(
            barmode='group',
            xaxis_title="Engine ID",
            yaxis_title="Remaining Useful Life (Cycles)",
            legend_title="Legend",
            template="plotly_dark",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- 2. Engine-Specific Analysis ---
    elif view_mode == "Engine-Specific Analysis":
        st.header("🔍 Individual Engine Diagnostics")
        
        engine_list = df_f['Engine_ID'].tolist()
        selected_engine = st.selectbox("Select an Engine ID to inspect:", engine_list)
        
        engine_data_f = df_f[df_f['Engine_ID'] == selected_engine].iloc[0]
        engine_data_c = df_c[df_c['Engine_ID'] == selected_engine].iloc[0]
        
        true_rul = engine_data_f['y_true']
        pred_f = engine_data_f['y_pred']
        pred_c = engine_data_c['y_pred']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("True RUL", f"{true_rul:.0f} cycles")
        col2.metric("Federated Prediction", f"{pred_f:.1f} cycles", delta=f"{pred_f - true_rul:.1f} error", delta_color="inverse")
        col3.metric("Centralized Prediction", f"{pred_c:.1f} cycles", delta=f"{pred_c - true_rul:.1f} error", delta_color="inverse")
        
        # Gauge Chart for Engine Health
        st.subheader("Engine Health Status")
        
        health_color = "green" if pred_f > 50 else ("orange" if pred_f > 20 else "red")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred_f,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Estimated Remaining Cycles"},
            gauge = {
                'axis': {'range': [0, 130]},
                'bar': {'color': health_color},
                'steps' : [
                    {'range': [0, 20], 'color': "rgba(255,0,0,0.3)"},
                    {'range': [20, 50], 'color': "rgba(255,165,0,0.3)"},
                    {'range': [50, 130], 'color': "rgba(0,128,0,0.3)"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 20}
            }
        ))
        
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # --- 3. Model Comparison Scatter ---
    elif view_mode == "Model Comparison Scatter":
        st.header("📈 Prediction Accuracy Scatter")
        st.markdown("A perfect model would have all dots directly on the red diagonal line.")
        
        fig = go.Figure()
        
        # Federated Scatter
        fig.add_trace(go.Scatter(
            x=df_f['y_true'], 
            y=df_f['y_pred'],
            mode='markers',
            name='Federated Model',
            marker=dict(color='#F39C12', size=8, opacity=0.7)
        ))
        
        # Centralized Scatter
        fig.add_trace(go.Scatter(
            x=df_c['y_true'], 
            y=df_c['y_pred'],
            mode='markers',
            name='Centralized Model',
            marker=dict(color='#4A90E2', size=8, opacity=0.7)
        ))
        
        # Ideal Line
        fig.add_trace(go.Scatter(
            x=[0, 130], 
            y=[0, 130],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title="True RUL (Cycles)",
            yaxis_title="Predicted RUL (Cycles)",
            template="plotly_dark",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
