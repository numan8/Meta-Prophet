import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import streamlit as st

# --- Load Excel ---
df = pd.read_excel("SLC_Pickup_Stops_7-11.xlsx")
df['StopDate'] = pd.to_datetime(df['StopDate'])
df = df[df['StopDate'].dt.dayofweek < 5]  # Weekdays only
df['HolidayFlag'] = (df['HolidayFlag'] == 'Yes').astype(int)

# --- Prepare for Prophet ---
df_prophet = df.rename(columns={'StopDate': 'ds', 'NumberOfPickupStops': 'y'})
model = Prophet(interval_width=0.95)
model.add_regressor('HolidayFlag')
model.fit(df_prophet[['ds', 'y', 'HolidayFlag']])

# --- Future Data ---
future = model.make_future_dataframe(periods=14, freq='B')
holiday_flags = df_prophet[['ds', 'HolidayFlag']].copy()
future = future.merge(holiday_flags, on='ds', how='left')
future['HolidayFlag'] = future['HolidayFlag'].fillna(0)
known_future_holidays = ['2025-08-14', '2025-12-25']
future.loc[future['ds'].isin(pd.to_datetime(known_future_holidays)), 'HolidayFlag'] = 1

# --- Forecast ---
forecast = model.predict(future)
forecast = forecast.merge(future[['ds', 'HolidayFlag']], on='ds', how='left')
forecast.loc[forecast['HolidayFlag'] == 1, ['yhat', 'yhat_lower', 'yhat_upper']] = 0

# --- Merge Actual + Forecast for Display ---
df_display = df_prophet.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='outer')
df_display = df_display.rename(columns={
    'ds': 'StopDate',
    'yhat': 'Forecasted',
    'yhat_lower': 'Lower Interval',
    'yhat_upper': 'Upper Interval',
    'y': 'NumberOfPickupStops'
})
df_display['HolidayFlag'] = df_display['HolidayFlag'].fillna(0).astype(int)
df_display = df_display.sort_values('StopDate')

# --- Streamlit UI ---
st.title("Pickup Stops Forecast (with Holiday Effect)")

# Forecast plot
st.subheader("Forecast Graph (95% CI)")
fig = model.plot(forecast)
st.pyplot(fig)

# Full scrollable table
st.subheader("Detailed Table (Actual + Forecasted)")
st.dataframe(df_display, use_container_width=True)

# Optional: download CSV
csv = df_display.to_csv(index=False).encode('utf-8')
st.download_button("Download Forecast CSV", csv, "forecast_output.csv", "text/csv")
