import streamlit as st
import os
import sys

# Set up path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Path to the actual dashboard
dashboard_path = os.path.join(current_dir, "dashboard", "dashboard_app_v3.py")

# Execute the dashboard code
if os.path.exists(dashboard_path):
    with open(dashboard_path, encoding='utf-8') as f:
        exec(f.read())
else:
    st.error("Dashboard file not found!")
