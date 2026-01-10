import streamlit as st
import pandas as pd
import json
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page Config
st.set_page_config(
    page_title="AI IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants & Config ---
LOG_FILE = "logs/alerts.jsonl"
REFRESH_RATE = 5 # seconds

# --- Helper Functions ---
@st.cache_data(ttl=5) # Cache data for 5 seconds
def load_data(log_file):
    if not os.path.exists(log_file):
        return pd.DataFrame()
    
    data = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        st.error(f"Error reading logs: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    
    # Preprocessing
    # Convert timestamp (assuming it might be float string or ISO)
    # Check format. In pipeline we stored str(df.iloc[i]['timestamp']) which is likely float seconds as string
    try:
        df['datetime'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
    except:
        # Fallback if ISO format
        df['datetime'] = pd.to_datetime(df['timestamp'])
        
    df['hour'] = df['datetime'].dt.hour
    df = df.sort_values(by='datetime', ascending=False)
    
    return df

# --- Sidebar ---
st.sidebar.title("🛡️ IDS Control")
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh Rate (s)", 1, 60, 5)

st.sidebar.subheader("Filters")
# Date Filter will be dynamic based on data
# Initialize placeholders if no data
df = load_data(LOG_FILE)

if not df.empty:
    min_date = df['datetime'].min().date()
    max_date = df['datetime'].max().date()
    
    start_date = st.sidebar.date_input("Start Date", min_date)
    end_date = st.sidebar.date_input("End Date", max_date)
    
    selected_levels = st.sidebar.multiselect(
        "Alert Level", 
        options=df['alert_level'].unique(),
        default=df['alert_level'].unique()
    )
    
    # Apply Filters
    mask = (df['datetime'].dt.date >= start_date) & \
           (df['datetime'].dt.date <= end_date) & \
           (df['alert_level'].isin(selected_levels))
    
    filtered_df = df.loc[mask]
else:
    st.sidebar.warning("No Data Found")
    filtered_df = pd.DataFrame()

# --- Main Dashboard ---
st.title("🛡️ AI Real-time Intrusion Detection Dashboard")
st.markdown(f"**Data Source**: `{LOG_FILE}` | **Last Update**: {datetime.now().strftime('%H:%M:%S')}")

if filtered_df.empty:
    st.info("Waiting for alerts... Run `python scripts/run_pipeline.py` to generate traffic.")
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
else:
    # 1. KPIs
    st.markdown("### 📊 Key Metrics (Selected Range)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    total_alerts = len(filtered_df)
    unique_src = filtered_df['src'].nunique()
    critical_count = len(filtered_df[filtered_df['alert_level'] == 'CRITICAL'])
    
    # Top Attack Type
    top_attack = "None"
    if 'attack_type' in filtered_df.columns and not filtered_df.empty:
        # Check if 'reasons' or 'features' gives hint, technically pipeline logs 'final_label'
        # Pipeline output currently doesn't strictly have 'attack_type' column, 
        # it has 'reasons' and 'final_label'. 
        # Let's see what we logged. Pipeline: 'reasons' contains text.
        # We can simulate attack type from reasons or label.
        # For now, let's use 'reasons' as proxy or risk_score
        top_attack = "N/A"
        
    kpi1.metric("Total Alerts", total_alerts)
    kpi2.metric("Suspicious IPs", unique_src)
    kpi3.metric("Critical Threats", critical_count, delta_color="inverse")
    
    # Risk Score Avg
    avg_risk = filtered_df['risk_score'].mean()
    kpi4.metric("Avg Risk Score", f"{avg_risk:.1f}/100")

    st.markdown("---")

    # 2. Charts
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("📈 Alert Frequency")
        # Resample to count by time bucket (e.g. Hour or Minute)
        # Using 10 minute buckets for view
        ts_df = filtered_df.set_index('datetime').resample('10min').size().reset_index(name='count')
        fig_time = px.area(ts_df, x='datetime', y='count', title="Alerts over Time (10min intervals)",
                           markers=True, color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig_time, use_container_width=True)
        
    with c2:
        st.subheader("🔴 Severity Distribution")
        pie_df = filtered_df['alert_level'].value_counts().reset_index()
        pie_df.columns = ['Level', 'Count']
        
        # Color map
        color_map = {
            'LOW': '#00CC96', 'MEDIUM': '#FFA15A', 
            'HIGH': '#EF553B', 'CRITICAL': '#B33DC6'
        }
        
        fig_pie = px.pie(pie_df, values='Count', names='Level', 
                         color='Level', color_discrete_map=color_map,
                         title="Alert Severity Split")
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    # 3. Tables & Details
    c3, c4 = st.columns([1, 1])
    with c3:
        st.subheader("🌐 Top Source IPs")
        top_src = filtered_df['src'].value_counts().head(10).reset_index()
        top_src.columns = ['Source IP', 'Alert Count']
        st.dataframe(top_src, use_container_width=True)
        
    with c4:
        st.subheader("📋 Top Target IPs")
        top_dst = filtered_df['dst'].value_counts().head(10).reset_index()
        top_dst.columns = ['Dest IP', 'Alert Count']
        st.dataframe(top_dst, use_container_width=True)

    st.markdown("---")
    st.subheader("🔬 Alert Timeline & Analysis")
    
    # Dataframe with basic cols
    disp_cols = ['datetime', 'src', 'dst', 'proto', 'alert_level', 'risk_score', 'reasons']
    # Filter available cols
    disp_cols = [c for c in disp_cols if c in filtered_df.columns]
    
    # Selection
    selected_event = st.dataframe(
        filtered_df[disp_cols], 
        use_container_width=True,
        selection_mode="single-row",
        on_select="rerun"
    )
    
    # Note: Streamlit recent versions support on_select. 
    # If using older version, might need st.session_state hack or different widget.
    # Assuming user has decent streamlit updated via requirements.
    
    # Since we can't easily get the selected row index from standard dataframe in all versions easily without callback,
    # let's use an expander for details if they want to dig deep, or just list recent 5 details.
    
    # Alternative: Selectbox for specific alert ID/Timestamp
    st.markdown("### 🔍 Drill Down")
    
    # Create valid options (Index + Time + Src)
    filtered_df['label'] = filtered_df['datetime'].astype(str) + " | " + filtered_df['src'] + " -> " + filtered_df['dst']
    alert_choice = st.selectbox("Select Alert to Inspect:", filtered_df['label'].tolist())
    
    if alert_choice:
        row = filtered_df[filtered_df['label'] == alert_choice].iloc[0]
        
        d1, d2 = st.columns(2)
        with d1:
            st.info(f"**timestamp**: {row['datetime']}")
            st.write(f"**Source**: {row['src']}")
            st.write(f"**Destination**: {row['dst']}")
            st.write(f"**Protocol**: {row['proto']}")
        
        with d2:
            lvl_color = "red" if row['alert_level'] in ['HIGH', 'CRITICAL'] else "orange"
            st.markdown(f"**Level**: :{lvl_color}[{row['alert_level']}]")
            st.metric("Risk Score", f"{row['risk_score']:.1f}")
            st.write(f"**Reason**: {row['reasons']}")
            
        # SHAP / Explanation
        # If pipeline saved exact explanations in log, we show them. 
        # Current pipeline saves 'reasons' string. 
        # If we want SHAP JSON, we need to have logged it.
        # Future improvement: Log full SHAP json in a separate field in pipeline.
        
        st.markdown("#### 🧠 AI Explanation")
        if 'explanation' in row and row['explanation']:
             st.json(row['explanation'])
        else:
             st.caption("No specific SHAP explanation payload available in this log entry.")

    # Auto Refresh Logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
