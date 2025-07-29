import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import streamlit as st

# --- App Title ---
st.set_page_config(layout="wide")
st.title("📦 LTL Pickup Forecast (with Holidays)")

# --- Load Excel File ---
file_path = "SLC_Pickup_Stops_7-11.xlsx"
df = pd.read_excel(file_path)

# --- Preprocessing ---
df['StopDate'] = pd.to_datetime(df['StopDate'])
df = df[df['StopDate'].dt.dayofweek < 5]  # Keep only weekdays
df['HolidayFlag'] = (df['HolidayFlag'] == 'Yes').astype(int)

# Prepare for Prophet
df_prophet = df.rename(columns={
    'StopDate': 'ds',
    'NumberOfPickupStops': 'y'
})[['ds', 'y', 'HolidayFlag']]

# --- Prophet Model ---
model = Prophet(interval_width=0.95)
model.add_regressor('HolidayFlag')
model.fit(df_prophet)

# --- Forecasting ---
future = model.make_future_dataframe(periods=14, freq='B')  # 14 business days
future = future.merge(df_prophet[['ds', 'HolidayFlag']], on='ds', how='left')
future['HolidayFlag'] = future['HolidayFlag'].fillna(0)

# Optional: Add known future holidays manually
known_future_holidays = ['2025-08-14', '2025-12-25']
future.loc[future['ds'].isin(pd.to_datetime(known_future_holidays)), 'HolidayFlag'] = 1

# --- Predict ---
forecast = model.predict(future)

# --- Merge HolidayFlag into forecast safely ---
if 'HolidayFlag' not in forecast.columns:
    forecast = forecast.merge(future[['ds', 'HolidayFlag']], on='ds', how='left')

# --- Force yhat = 0 on holidays ---
forecast.loc[forecast['HolidayFlag'] == 1, ['yhat', 'yhat_lower', 'yhat_upper']] = 0

# --- Plot Forecast ---
st.subheader("📈 Forecast Plot (95% CI)")
fig = model.plot(forecast)
st.pyplot(fig)

# --- Combine Actual + Forecast for Display ---
df_display = df_prophet.merge(
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='outer'
).rename(columns={
    'ds': 'StopDate',
    'yhat': 'Forecasted',
    'yhat_lower': 'Lower Interval',
    'yhat_upper': 'Upper Interval',
    'y': 'NumberOfPickupStops'
})
df_display['HolidayFlag'] = df_display['HolidayFlag'].fillna(0).astype(int)
df_display = df_display.sort_values('StopDate')

# --- Table View ---
st.subheader("📋 Detailed Forecast Table")
st.dataframe(df_display, use_container_width=True)

# --- Download Button ---
csv = df_display.to_csv(index=False).encode('utf-8')
st.download_button(
    "⬇️ Download Forecast as CSV",
    csv,
    "ltl_forecast.csv",
    "text/csv",
    key='download-csv'
)
