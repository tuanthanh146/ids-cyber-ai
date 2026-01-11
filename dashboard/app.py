import streamlit as st
import pandas as pd
import json
import os
import sys
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

# Try to import LabelManager, handle if missing
try:
    from utils.label_manager import LabelManager
except ImportError:
    LabelManager = None

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
    try:
        df['datetime'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
    except:
        try:
            df['datetime'] = pd.to_datetime(df['timestamp'])
        except:
            df['datetime'] = datetime.now() # Fallback
        
    if 'datetime' in df.columns:
        df['hour'] = df['datetime'].dt.hour
        df = df.sort_values(by='datetime', ascending=False)
    
    return df

# --- Sidebar Global ---
st.sidebar.title("🛡️ IDS Control")
mode = st.sidebar.radio("View Mode", ["Live Dashboard", "Active Learning (Labeling)"])
st.sidebar.markdown("---")

# ==========================================
# MODE: LIVE DASHBOARD (ORIGINAL)
# ==========================================
if mode == "Live Dashboard":
    # --- Sidebar Controls for Dashboard ---
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Rate (s)", 1, 60, 5)

    st.sidebar.subheader("Filters")
    
    df = load_data(LOG_FILE)

    if not df.empty:
        min_date = df['datetime'].min().date()
        max_date = df['datetime'].max().date()
        
        start_date = st.sidebar.date_input("Start Date", min_date)
        end_date = st.sidebar.date_input("End Date", max_date)
        
        options = df['alert_level'].unique() if 'alert_level' in df.columns else []
        selected_levels = st.sidebar.multiselect(
            "Alert Level", 
            options=options,
            default=options
        )
        
        # Apply Filters
        mask = (df['datetime'].dt.date >= start_date) & \
               (df['datetime'].dt.date <= end_date)
        
        if 'alert_level' in df.columns:
            mask = mask & (df['alert_level'].isin(selected_levels))
               
        filtered_df = df.loc[mask]
    else:
        st.sidebar.warning("No Data Found")
        filtered_df = pd.DataFrame()

    # --- Main Dashboard Content ---
    st.title("🛡️ AI Real-time Intrusion Detection Dashboard")
    st.markdown(f"**Data Source**: `{LOG_FILE}` | **Last Update**: {datetime.now().strftime('%H:%M:%S')}")

    if filtered_df.empty:
        st.info("Waiting for alerts... Run `python scripts/run_pipeline.py` to generate traffic.")
    else:
        # 1. KPIs
        st.markdown("### 📊 Key Metrics (Selected Range)")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        total_alerts = len(filtered_df)
        unique_src = filtered_df['src'].nunique() if 'src' in filtered_df.columns else 0
        critical_count = len(filtered_df[filtered_df['alert_level'] == 'CRITICAL']) if 'alert_level' in filtered_df.columns else 0
        
        kpi1.metric("Total Alerts", total_alerts)
        kpi2.metric("Suspicious IPs", unique_src)
        kpi3.metric("Critical Threats", critical_count, delta_color="inverse")
        
        # Risk Score Avg
        avg_risk = filtered_df['risk_score'].mean() if 'risk_score' in filtered_df.columns else 0
        kpi4.metric("Avg Risk Score", f"{avg_risk:.1f}/100")

        st.markdown("---")

        # 2. Charts
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("📈 Alert Frequency")
            if 'datetime' in filtered_df.columns:
                ts_df = filtered_df.set_index('datetime').resample('10min').size().reset_index(name='count')
                fig_time = px.area(ts_df, x='datetime', y='count', title="Alerts over Time (10min intervals)",
                                   markers=True, color_discrete_sequence=['#FF4B4B'])
                st.plotly_chart(fig_time, use_container_width=True)
            
        with c2:
            st.subheader("🔴 Severity Distribution")
            if 'alert_level' in filtered_df.columns:
                pie_df = filtered_df['alert_level'].value_counts().reset_index()
                pie_df.columns = ['Level', 'Count']
                
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
            if 'src' in filtered_df.columns:
                top_src = filtered_df['src'].value_counts().head(10).reset_index()
                top_src.columns = ['Source IP', 'Alert Count']
                st.dataframe(top_src, use_container_width=True)
            
        with c4:
            st.subheader("📋 Top Target IPs")
            if 'dst' in filtered_df.columns:
                top_dst = filtered_df['dst'].value_counts().head(10).reset_index()
                top_dst.columns = ['Dest IP', 'Alert Count']
                st.dataframe(top_dst, use_container_width=True)

        st.markdown("---")
        st.subheader("🔬 Alert Timeline & Analysis")
        
        disp_cols = ['datetime', 'src', 'dst', 'proto', 'alert_level', 'risk_score', 'reasons']
        disp_cols = [c for c in disp_cols if c in filtered_df.columns]
        
        # Drill down logic
        filtered_df['label'] = filtered_df['datetime'].astype(str) + " | " + filtered_df['src'].astype(str) + " -> " + filtered_df['dst'].astype(str)
        alert_choice = st.selectbox("Select Alert to Inspect:", filtered_df['label'].tolist())
        
        if alert_choice:
            row = filtered_df[filtered_df['label'] == alert_choice].iloc[0]
            
            d1, d2 = st.columns(2)
            with d1:
                st.info(f"**timestamp**: {row.get('datetime')}")
                st.write(f"**Source**: {row.get('src')}")
                st.write(f"**Destination**: {row.get('dst')}")
                st.write(f"**Protocol**: {row.get('proto')}")
            
            with d2:
                lvl = row.get('alert_level', 'UNKNOWN')
                lvl_color = "red" if lvl in ['HIGH', 'CRITICAL'] else "orange"
                st.markdown(f"**Level**: :{lvl_color}[{lvl}]")
                st.metric("Risk Score", f"{row.get('risk_score', 0):.1f}")
                st.write(f"**Reason**: {row.get('reasons')}")
                
            st.markdown("#### 🧠 AI Explanation")
            if 'explanation' in row and row['explanation']:
                 st.json(row['explanation'])
            else:
                 st.caption("No specific SHAP explanation available.")

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

# ==========================================
# MODE: ACTIVE LEARNING (NEW)
# ==========================================
elif mode == "Active Learning (Labeling)":
    st.title("🧠 Active Learning & Labeling")
    st.markdown("Review 'Hard Samples' to improve AI accuracy.")
    
    if LabelManager:
        lm = LabelManager()
        all_data = lm.load_data()
        
        if all_data.empty:
            st.warning("No data in Label Store.")
            if st.button("📥 Import Recent Alerts to Label Store"):
                alerts = load_data(LOG_FILE)
                if not alerts.empty:
                    # Enrich with necessary fields for labeling
                    if 'risk_score' in alerts.columns:
                        alerts['model_score'] = alerts['risk_score'] / 100.0
                    else:
                        alerts['model_score'] = 0.5
                    
                    # Generate flow_id if missing (Mock)
                    alerts['flow_id'] = alerts['datetime'].astype(str) + "_" + alerts.index.astype(str)
                    
                    alerts['label'] = None
                    alerts['label_status'] = 'UNLABELED'
                    
                    lm.save_data(alerts)
                    st.success(f"Imported {len(alerts)} alerts. Refreshing...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Log file is empty.")
        
        # Stats
        if not all_data.empty:
            st.markdown("### 📊 Labeling Stats")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Flow", len(all_data))
            c2.metric("Verified", len(all_data[all_data['label_status']=='VERIFIED']))
            c3.metric("Pseudo-Labeled", len(all_data[all_data['label_status']=='PSEUDO_LABELED']))
            c4.metric("Pending Review", len(all_data[all_data['label_status']=='HUMAN_REVIEW_NEEDED']))
            
            st.markdown("---")
            
            # Review Section
            st.subheader("🔍 Hard Sample Review")
            
            # Filter Logic
            filter_status = st.selectbox("Filter Status", ["HUMAN_REVIEW_NEEDED", "UNLABELED", "PSEUDO_LABELED", "VERIFIED"])
            
            # If HUMAN_REVIEW_NEEDED is empty, optionally offer to run heuristic selection
            if filter_status == "HUMAN_REVIEW_NEEDED":
                 subset = all_data[all_data['label_status'] == 'HUMAN_REVIEW_NEEDED']
                 if subset.empty:
                     st.info("No hard samples currently flagged.")
                     if st.button("Run Heuristic Selection (Find Hard Samples)"):
                         subset = lm.select_hard_samples(all_data)
                         st.rerun()
            else:
                 subset = all_data[all_data['label_status'] == filter_status]
            
            if subset.empty:
                st.info(f"No samples found with status: {filter_status}")
            else:
                # Pagination or simple list
                for idx, row in subset.head(10).iterrows():
                    with st.expander(f"Review Flow: {row.get('src', '?')} -> {row.get('dst', '?')} (Score: {row.get('model_score', 0):.2f})", expanded=True):
                        cA, cB = st.columns([2, 1])
                        
                        with cA:
                            st.write(f"**Proto**: {row.get('proto')} | **Reasons**: {row.get('reasons')}")
                            st.write(f"**Current Status**: `{row.get('label_status')}`")
                            st.write(f"**Current Label**: `{row.get('label')}`")
                            if 'explanation' in row:
                                st.json(row['explanation'])
                                
                        with cB:
                            new_label = st.selectbox("Assign Label", ["NORMAL", "ReviewLater", "DoS", "PortScan", "BruteForce", "Malware"], key=f"lbl_{idx}")
                            if st.button("Confirm Label", key=f"btn_{idx}"):
                                lm.update_labels([row['flow_id']], new_label, "admin_dashboard")
                                st.success("Label Updated!")
                                time.sleep(0.5)
                                st.rerun()
            
            st.markdown("---")
            if st.button("🤖 Trigger Auto-Labeling (Pseudo-Labels)"):
                with st.spinner("Applying rules..."):
                    res = lm.auto_label(all_data[all_data['label_status']=='UNLABELED'])
                    # Save back handled inside auto_label? No, auto_label returns logical list but we need to save.
                    # Actually utils/label_manager.py auto_label returns DATAFRAME.
                    # We should merge. For simplicity in this demo, let's just save valid results.
                    # Re-implementation note: LabelManager.auto_label() in previous step returns DF but didn't save. 
                    # We need to save manually or update the function. 
                    # Let's just save 'res' for now roughly overwriting or merging.
                    # Ideally: lm.save_data(res)
                    # Merging logic:
                    if not res.empty:
                        # Update all_data with res rows
                        # Basic ID match...
                        all_data_dict = all_data.set_index('flow_id').to_dict('index')
                        res_dict = res.set_index('flow_id').to_dict('index')
                        all_data_dict.update(res_dict)
                        final_df = pd.DataFrame.from_dict(all_data_dict, orient='index').reset_index().rename(columns={'index':'flow_id'})
                        lm.save_data(final_df)
                        st.success("Auto-labeling complete!")
                        st.rerun()
                    else:
                        st.warning("No unlabeled data to process.")

    else:
        st.error("LabelManager module not found. Check utils/label_manager.py")
