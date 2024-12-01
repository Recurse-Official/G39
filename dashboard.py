import streamlit as st
from streamlit_echarts import st_echarts
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random

# Set page configuration
st.set_page_config(page_title="SyncTech Dashboard", layout="wide")

# Custom styles
st.markdown("""
    <style>
        .main-container {
            background-color: #f8f9fa;
        }
        .metric-box {
            border-radius: 10px;
            padding: 20px;
            color: white;
            margin: 10px 0;
            text-align: center;
        }
        .blue { background-color: #1e88e5; }
        .green { background-color: #28a745; }
        .red { background-color: #dc3545; }
        .dark-blue { background-color: #2c3e50; }
    </style>
""", unsafe_allow_html=True)

# Navigation
menu = ["Dashboard", "Employee Profile", "Live Camera Feed", "Employee Metrics", "Report System"]
choice = st.sidebar.selectbox("Menu", menu)

# Generate dummy data for visualizations
np.random.seed(42)
hours = list(range(8, 21))
persons_in_queue = np.random.randint(5, 50, size=len(hours))
avg_waiting_times = np.random.randint(5, 20, size=len(hours))
counter_efficiency = np.random.uniform(80, 95, size=len(hours))

# DataFrame for queue insights
queue_df = pd.DataFrame({
    "Hour": hours,
    "Persons in Queue": persons_in_queue,
    "Average Waiting Time (mins)": avg_waiting_times,
    "Counter Efficiency (%)": counter_efficiency
})

# Dashboard content
if choice == "Dashboard":
    st.markdown("<h1 style='text-align: center;'>SyncTech Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("### Real-Time Analytics and Visualizations")

    # Line chart: Queue Insights
    fig1 = px.line(queue_df, x="Hour", y="Persons in Queue", title="Queue Size Throughout the Day", markers=True)
    st.plotly_chart(fig1, use_container_width=True)

    # Bar chart: Average Waiting Time
    fig2 = px.bar(queue_df, x="Hour", y="Average Waiting Time (mins)", title="Average Waiting Time per Hour",
                  color="Average Waiting Time (mins)", color_continuous_scale="Viridis")
    st.plotly_chart(fig2, use_container_width=True)

    # Gauge chart: Counter Efficiency
    avg_efficiency = counter_efficiency.mean()
    fig3 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_efficiency,
        title={"text": "Average Counter Efficiency (%)"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "green"}}
    ))
    st.plotly_chart(fig3, use_container_width=True)

    # Insights
    st.markdown("### Insights")
    st.write(f"- Peak queue size observed at hour **{queue_df.loc[queue_df['Persons in Queue'].idxmax(), 'Hour']}**.")
    st.write(f"- Average waiting time across the day is **{queue_df['Average Waiting Time (mins)'].mean():.2f} mins**.")
    st.write(f"- Overall counter efficiency stands at **{avg_efficiency:.2f}%**.")

    # Optimization Tips
    st.markdown("### Queue Management Tips")
    tips = [
        "Schedule additional staff during peak hours to reduce waiting times.",
        "Implement dynamic queue allocation to improve counter efficiency.",
        "Encourage self-service kiosks during high traffic periods.",
        "Optimize employee shifts based on hourly traffic data.",
        "Use emotion detection to identify and resolve customer frustrations proactively."
    ]
    st.write(random.choice(tips))

elif choice == "Employee Profile":
    st.markdown("<h1>Employee Profiles</h1>", unsafe_allow_html=True)
    # Sample employee data
    employee_data = pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie", "Diana"],
        "Designation": ["Manager", "Support", "Clerk", "Supervisor"],
    })
    for _, row in employee_data.iterrows():
        st.markdown(f"<div class='metric-box dark-blue'>"
                    f"<h3>{row['Name']}</h3>"
                    f"<p>{row['Designation']}</p>"
                    "</div>", unsafe_allow_html=True)

elif choice == "Live Camera Feed":
    st.markdown("<h1>Live Camera Feed</h1>", unsafe_allow_html=True)
    st.markdown("Real-time feeds will be displayed here.")

elif choice == "Employee Metrics":
    st.markdown("<h1>Employee Metrics</h1>", unsafe_allow_html=True)
    # Example employee metrics
    metrics_df = pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie", "Diana"],
        "Tasks Completed": [25, 20, 22, 24],
        "Efficiency (%)": [88, 75, 80, 85],
        "Customer Satisfaction (%)": [95, 80, 85, 90]
    })
    st.dataframe(metrics_df)

elif choice == "Report System":
    st.markdown("<h1>Report System</h1>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload photos/videos of issues", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            st.write(f"Uploaded file: {file.name}")
