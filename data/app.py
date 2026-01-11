import streamlit as st
import pandas as pd
import time 
import os
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# 1. Setup Time and Date FIRST
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

# 2. Setup Auto-refresh
count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

# 3. Streamlit UI Header
st.title("Attendance Dashboard")

if count == 0:
    st.write("Refreshing data...")
else:
    st.write(f"Last update: {timestamp}")

# 4. Safe File Loading Logic
path = "Attendance/Attendance_" + date + ".csv"

if os.path.exists(path):
    # If the file exists, read and display it
    df = pd.read_csv(path)
    st.dataframe(df.style.highlight_max(axis=0))
else:
    # If the file is missing, show a warning instead of crashing
    st.warning(f"No attendance record found for today ({date}) yet.")
    st.info("Please run 'test.py' to begin taking attendance.")