import streamlit as st
import pandas as pd
import json
import os
import sys
import time
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

# Try imports
try:
    from utils.label_manager import LabelManager
except ImportError:
    LabelManager = None

# --- Configuration ---
st.set_page_config(
    page_title="SIEM Dashboard v3",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "The Look"
st.markdown("""
<style>
    /* Dark Theme enhancement */
    .stApp {
        background-color: #0E1117;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #363945;
        text-align: center;
    }
    /* Typography */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        color: #FAFAFA;
    }
    /* Charts background */
    .js-plotly-plot .plotly .main-svg {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
# --- Constants ---
LOG_FILE = "logs/alerts.jsonl"
if not os.path.exists(LOG_FILE):
    # Fallback for Streamlit Cloud or when logs aren't present
    LOG_FILE = "data/sample_alerts.jsonl"

# --- Helper Functions ---
@st.cache_data(ttl=5)
def load_data(log_file, limit_rows=5000):
    if not os.path.exists(log_file):
        return pd.DataFrame()
    
    data = []
    # Read last N lines efficiently for large files? 
    # For now, standard read
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Taking last N lines
            lines = lines[-limit_rows:] if len(lines) > limit_rows else lines
            for line in lines:
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
    
    # Standardize Timestamp
    if 'timestamp' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
        except:
            df['datetime'] = pd.to_datetime(df['timestamp'])
            
    if 'datetime' not in df.columns:
        df['datetime'] = datetime.now()
        
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df = df.sort_values(by='datetime', ascending=False)
    
    return df

def render_network_graph(df):
    if df.empty or 'src' not in df.columns or 'dst' not in df.columns:
        st.warning("Not enough data for Network Map.")
        return

    # Build Graph
    G = nx.DiGraph()
    
    # Aggregate flows
    flows = df.groupby(['src', 'dst']).size().reset_index(name='count')
    # Limit to top 50 flows for performance
    flows = flows.sort_values('count', ascending=False).head(50)
    
    for _, row in flows.iterrows():
        G.add_edge(row['src'], row['dst'], weight=row['count'])
        
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        # Size by degree
        node_size.append(10 + G.degree(node)*2)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="bottom center",
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            reversescale=True,
            color=[],
            size=node_size,
            colorbar=dict(
                thickness=15,
                title='Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
            
    # Color by connections
    node_adjacencies = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
    node_trace.marker.color = node_adjacencies

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='Use scroll to zoom',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    st.plotly_chart(fig, use_container_width=True)

# --- Sidebar ---
st.sidebar.title("🛡️ SIEM Dashboard v3")
st.sidebar.markdown("Advanced Threat Monitoring")

# Config
limit_rows = st.sidebar.number_input("Limit Rows", 1000, 100000, 5000)
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
if auto_refresh:
    refresh_rate = st.sidebar.slider("Refresh (s)", 2, 60, 5)

st.sidebar.divider()

# Load Data
df = load_data(LOG_FILE, limit_rows)

if df.empty:
    st.sidebar.warning("No Data Available")
else:
    min_time = df['datetime'].min().strftime('%Y-%m-%d %H:%M')
    max_time = df['datetime'].max().strftime('%Y-%m-%d %H:%M')
    st.sidebar.info(f"📅 Data Range:\n{min_time}\n{max_time}")

# --- Main Tabs ---
tab1, tab4, tab5 = st.tabs([
    "📊 Overview & Analytics", 
    "📝 Log Explorer",
    "🧠 Active Learning"
])

# ================= TAB 1: OVERVIEW (Classic Style) =================
with tab1:
    st.markdown(f"**Status**: Online | **Last Update**: {datetime.now().strftime('%H:%M:%S')}")
    
    if not df.empty:
        # 1. KPIs
        k1, k2, k3, k4 = st.columns(4)
        total_alerts = len(df)
        suspicious_ips = df['src'].nunique() if 'src' in df.columns else 0
        critical_alerts = len(df[df['alert_level'] == 'CRITICAL']) if 'alert_level' in df.columns else 0
        avg_risk = df['risk_score'].mean() if 'risk_score' in df.columns else 0
        
        k1.metric("Total Alerts", total_alerts)
        k2.metric("Suspicious IPs", suspicious_ips)
        k3.metric("Critical Threats", critical_alerts, delta_color="inverse")
        k4.metric("Avg Risk Score", f"{avg_risk:.1f}")

        st.divider()

        # Threat Gauge
        st.subheader("🔥 Threat Level Gauge")
        # Calculate Gauge Value (e.g., avg risk score or ratio of high alerts)
        gauge_val = df['risk_score'].mean() if 'risk_score' in df.columns else 0
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = gauge_val,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Avg Risk Score"},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "darkred"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#00CC96'},
                    {'range': [30, 70], 'color': '#FFA15A'},
                    {'range': [70, 100], 'color': '#EF553B'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.divider()
        
        # 2. Charts
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Alert Frequency (10min)")
            ts_df = df.set_index('datetime').resample('10min').size().reset_index(name='count')
            fig_time = px.area(ts_df, x='datetime', y='count', 
                               color_discrete_sequence=['#FF4B4B'])
            fig_time.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_time, use_container_width=True)
            
        with c2:
            st.subheader("Severity Distribution")
            if 'alert_level' in df.columns:
                counts = df['alert_level'].value_counts()
                # FIX: Use px.pie instead of px.donut
                fig_pie = px.pie(values=counts.values, names=counts.index, hole=0.4,
                                   color_discrete_sequence=px.colors.sequential.RdBu)
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_pie, use_container_width=True)


        # 3. Top Statistics (Bar Charts)
        st.subheader("📊 Top Statistics (Top 10)")
        t1, t2 = st.columns(2)
        
        with t1:
            st.caption("Top Attacker IPs")
            if 'src' in df.columns:
                top_src = df['src'].value_counts().head(10).reset_index()
                top_src.columns = ['IP', 'Count']
                fig_src = px.bar(top_src, x='Count', y='IP', orientation='h', 
                                 title="Top 10 Attacker IPs", color='Count', 
                                 color_continuous_scale='Reds')
                fig_src.update_layout(yaxis={'categoryorder':'total ascending'}, 
                                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_src, use_container_width=True)
                
        with t2:
            st.caption("Top Target Ports")
            # Check for dst_port or dest_port
            port_col = 'dst_port' if 'dst_port' in df.columns else ('dest_port' if 'dest_port' in df.columns else None)
            if port_col:
                top_dst_port = df[port_col].value_counts().head(10).reset_index()
                top_dst_port.columns = ['Port', 'Count']
                # Ensure Port is treated as categorical for bar chart if it's numeric
                top_dst_port['Port'] = top_dst_port['Port'].astype(str)
                
                fig_port = px.bar(top_dst_port, x='Count', y='Port', orientation='h',
                                  title="Top 10 Target Ports", color='Count',
                                  color_continuous_scale='Blues')
                fig_port.update_layout(yaxis={'categoryorder':'total ascending'},
                                       paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_port, use_container_width=True)
            else:
                st.info("No 'dst_port' column found in logs.")

        st.divider()
        
        # 4. Model Signals & Risk Analysis
        st.subheader("📉 Model Signals & Risk Analysis")
        m1, m2 = st.columns(2)
        
        with m1:
             st.caption("Risk Score Histogram")
             if 'risk_score' in df.columns:
                 fig_hist = px.histogram(df, x="risk_score", nbins=20, 
                                         height=350,
                                         title="Risk Score Distribution",
                                         color_discrete_sequence=['#FFA15A'])
                 fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                 st.plotly_chart(fig_hist, use_container_width=True)
        
        with m2:
             st.caption("Model Signals (Anomaly vs XGB)")
             if 'anomaly_score' in df.columns and 'xgb_prob' in df.columns:
                 fig_scat = px.scatter(df, x="anomaly_score", y="xgb_prob", 
                                       color="alert_level", 
                                       size='risk_score',
                                       height=350,
                                       hover_data=['src', 'dst', 'final_label'],
                                       title="Anomaly Score vs XGB Probability")
                 fig_scat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                 st.plotly_chart(fig_scat, use_container_width=True)
             else:
                 st.info("Missing model score columns.")

        st.divider()
        
        # 5. Alert Timeline & Drilldown
        st.subheader("🔬 Alert Timeline & Analysis")
        
        # Build label for selectbox to match v2
        df['label_view'] = df.apply(lambda x: f"{x['datetime'].strftime('%H:%M:%S')} | {x.get('src','?')} -> {x.get('dst','?')} [{x.get('alert_level','?')}]", axis=1)
        
        slide_val = st.selectbox("Select Alert to Inspect", df['label_view'].tolist())
        
        if slide_val:
            row = df[df['label_view'] == slide_val].iloc[0]
            
            d1, d2, d3 = st.columns([1, 1, 2])
            with d1:
                st.info(f"**Timestamp**: {row['datetime']}")
                st.write(f"**Protocol**: {row.get('proto')}")
                st.write(f"**Attack**: {row.get('final_label')}")
            
            with d2:
                lvl = row.get('alert_level')
                color = "red" if lvl in ['HIGH', 'CRITICAL'] else "orange"
                st.markdown(f"**Level**: :{color}[{lvl}]")
                st.metric("Risk Score", f"{row.get('risk_score'):.1f}")
                
            with d3:
                st.markdown("#### 🧠 AI Explanation")
                expl = row.get('shap_explanation')
                if expl and isinstance(expl, list):
                    # Convert list of dicts to nice bullets or chart
                    # Assuming [{feature, value, effect}]
                    for feat in expl:
                        eff = "⬆️" if feat.get('impact') == 'Positive' else "⬇️"
                        st.write(f"{eff} **{feat.get('feature')}**: {feat.get('value')} (SHAP: {feat.get('shap_value'):.3f})")
                elif expl:
                     st.json(expl)
                else:
                    st.caption("No deep explanation available for this alert.")

        st.divider()
        
        # 5. Network Map
        st.subheader("🕸️ Network Attack Graph")
        if not df.empty:
            st.caption("Visualizing communication flows (Src -> Dst). Hover for details.")
            render_network_graph(df)
        else:
            st.info("No interactions to map.")

# ================= TAB 2: THREAT ANALYTICS (New) =================

        st.divider()
        
        # 7. Advanced Analytics (Heatmap Only - Others moved up)
        st.subheader("🔥 Activity Heatmap")
        st.caption("Alert Density (Hour x Minute)")
        heatmap_data = df.groupby(['hour', 'minute']).size().unstack(fill_value=0)
        fig_heat = px.imshow(heatmap_data, labels=dict(x="Minute", y="HourOfDay", color="Count"),
                             aspect="auto", color_continuous_scale="Viridis")
        fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_heat, use_container_width=True)



# ================= TAB 4: LOG EXPLORER (New) =================
with tab4:
    st.header("📝 Log Explorer")
    if not df.empty:
        # Filters
        f1, f2, f3 = st.columns(3)
        with f1:
            search_ip = st.text_input("Search IP")
        with f2:
            filter_lvl = st.multiselect("Filter Level", df['alert_level'].unique())
        with f3:
             filter_proto = st.multiselect("Filter Proto", df['proto'].unique())
             
        # Apply
        local_df = df.copy()
        if search_ip:
            local_df = local_df[local_df['src'].str.contains(search_ip) | local_df['dst'].str.contains(search_ip)]
        if filter_lvl:
            local_df = local_df[local_df['alert_level'].isin(filter_lvl)]
        if filter_proto:
            local_df = local_df[local_df['proto'].isin(filter_proto)]
            
        st.dataframe(local_df, use_container_width=True)
        st.caption(f"Showing {len(local_df)} rows")
        
        # CSV Export
        csv = local_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="ids_logs_export.csv", mime="text/csv")
    else:
        st.info("Log Empty")

# ================= TAB 5: ACTIVE LEARNING (Preserved) =================
with tab5:
    if LabelManager:
        # Reuse logic from v2 app (simplified import or direct code)
        # For Brevity, pointing user to run Labeling if needed or simple iframe?
        # Let's simple reimplement minimal Label view
        st.header("🧠 Active Learning Integration")
        
        lm = LabelManager()
        lab_df = lm.load_data()
        
        st.metric("Total Labeled/Tracked Flows", len(lab_df))
        
        # Hard Samples
        if 'label_status' in lab_df.columns:
            hard_samples = lab_df[lab_df['label_status'] == 'HUMAN_REVIEW_NEEDED']
            if not hard_samples.empty:
                st.warning(f"⚠️ {len(hard_samples)} Samples need Human Verification")
                cols = ['timestamp', 'src', 'dst', 'model_score', 'reasons']
                valid_cols = [c for c in cols if c in hard_samples.columns]
                st.dataframe(hard_samples[valid_cols])
            else:
                st.success("No pending hard samples.")
        else:
            st.info("No label status information found.")
            
        # Button import
        if st.button("Sync Logs to Label Store"):
             # Convert datetime objects to string for JSON serialization
             df_to_save = df.copy()
             if 'datetime' in df_to_save.columns:
                 df_to_save['datetime'] = df_to_save['datetime'].astype(str)
             
             lm.save_data(df_to_save) # Basic sync
             st.success("Synced!")
             
    else:
        st.error("LabelManager not found.")

# Auto Refresh Logic (Manual)
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
