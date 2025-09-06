import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load dataset
file_path = "carbon_impact_analysis.csv"  # Adjust path if needed
df = pd.read_csv(r'C:\Users\Alejandro\OneDrive\Documentos\Data Analyst Portfolio\ITAD Data Analysis Dashboard\DataSets\carbon_impact_analysis.csv')

st.set_page_config(page_title="Carbon Savings Dashboard", layout="wide")
st.title("🌍 Interactive Carbon Savings Dashboard")

# Sidebar filters
st.sidebar.header("Filters")
selected_region = st.sidebar.multiselect("Select Region", options=df["region"].unique(), default=df["region"].unique())
selected_device = st.sidebar.multiselect("Select Device Type", options=df["equipment_type"].unique(), default=df["equipment_type"].unique())

# Apply filters
df_filtered = df[(df["region"].isin(selected_region)) & (df["equipment_type"].isin(selected_device))]

# Aggregations
total_carbon_savings = df_filtered['carbon_savings_kg'].sum()/1000
device_savings = df_filtered.groupby('equipment_type')['carbon_savings_kg'].sum().sort_values(ascending=False)/1000
time_savings = df_filtered.groupby('lifecycle_extension_years')['carbon_savings_kg'].sum().sort_index()/1000
region_savings = df_filtered.groupby('region')['carbon_savings_kg'].sum().sort_values(ascending=False)/1000

# Layout with 2x2 charts
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Total Carbon Savings",
        "CO2e Avoided by Device Type",
        "Carbon Savings Over Time",
        "CO2e Avoided by Region"
    ),
    specs=[[{"type": "bar"}, {"type": "bar"}],
           [{"type": "scatter"}, {"type": "pie"}]]
)

# Total Carbon Savings
fig.add_trace(go.Bar(x=["Total CO2e Savings"], y=[total_carbon_savings], marker_color="seagreen"), row=1, col=1)

# CO2e by Device Type
fig.add_trace(go.Bar(x=device_savings.values, y=device_savings.index, orientation="h",
                     marker=dict(color=device_savings.values, colorscale="Viridis")), row=1, col=2)

# Carbon Savings Over Time
fig.add_trace(go.Scatter(x=time_savings.index, y=time_savings.values, mode="lines+markers", line=dict(color="darkblue")), row=2, col=1)

# CO2e by Region (Pie)
fig.add_trace(go.Pie(labels=region_savings.index, values=region_savings.values, hole=0.3), row=2, col=2)

# Update Layout
fig.update_layout(
    title_text="Interactive Carbon Savings Dashboard",
    height=800,
    showlegend=True
)

# Render dashboard
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Dataset Preview")
st.dataframe(df_filtered.head(20))
