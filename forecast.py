# # forecast_aapl.py
# from prophet import Prophet
# import pandas as pd
# import matplotlib.pyplot as plt

# # Step 1: Load your CSV
# df = pd.read_csv("/Users/afrahanas/Downloads/project/AAPL_all_data.csv")

# # Step 2: Prepare data for Prophet
# df = df.rename(columns={
#     'Date': 'ds',     # change 'Date' if your CSV uses another column name
#     'Close': 'y'      # change 'Close' if your CSV uses another column name
# })
# df['ds'] = pd.to_datetime(df['ds'])
# df = df.sort_values('ds')

# # Step 3: Create and fit the model
# model = Prophet(daily_seasonality=True, yearly_seasonality=True)
# model.fit(df)

# # Step 4: Make future predictions
# future = model.make_future_dataframe(periods=30)  # predict next 30 days
# forecast = model.predict(future)

# # Step 5: Print forecast
# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# # Step 6: Plot forecast
# model.plot(forecast)
# plt.title("AAPL Stock Price Forecast")
# plt.xlabel("Date")
# plt.ylabel("Price")
# plt.show()

# import pandas as pd

# df = pd.read_csv("/Users/afrahanas/Downloads/project/AAPL_all_data.csv")
# print(df.columns)


# Import libraries
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
# Load the dataset
# We skip the second and third rows as they contain metadata and not the actual data or headers.
df = pd.read_csv("/Users/afrahanas/Downloads/project3/AAPL_all_data.csv", header=0, skiprows=[1, 2])


# Rename the first column to 'Date'
df.rename(columns={'Price': 'Date'}, inplace=True)

# Create a new DataFrame for Prophet with the required column names 'ds' and 'y'
df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

# Convert the 'ds' column to a datetime object
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

# Display the first few rows of the prepared data
print(df_prophet.head())

# Create an instance of the Prophet model
model = Prophet()

# Fit the model to your data
model.fit(df_prophet)

# Create a dataframe for future predictions (forecasting for the next 365 days)
future = model.make_future_dataframe(periods=365)

# Generate the forecast
forecast = model.predict(future)

# Display the last few rows of the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
# Plot the forecast
# The black dots represent the actual data points.
# The blue line is the forecasted values (yhat).
# The light blue shaded area is the uncertainty interval.
fig1 = model.plot(forecast)
plt.title('Apple Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Plot the forecast components
# This will show you the trend, and seasonal patterns (yearly and weekly).
fig2 = model.plot_components(forecast)
plt.show()
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
import datetime as dt

# Get the last date in your training data
last_date = df_prophet['ds'].max()

# Filter forecast for future dates only
future_forecast = forecast[forecast['ds'] > last_date]

# 1️⃣ Tomorrow's forecast
tomorrow = last_date + dt.timedelta(days=1)
yhat_tomorrow = future_forecast[future_forecast['ds'] == tomorrow]['yhat'].values
if len(yhat_tomorrow) > 0:
    yhat_tomorrow = yhat_tomorrow[0]
else:
    yhat_tomorrow = None

# 2️⃣ Average of next week (7 days)
next_week = future_forecast[future_forecast['ds'] <= last_date + dt.timedelta(days=7)]
avg_next_week = next_week['yhat'].mean()

# 3️⃣ Average of next month (30 days)
next_month = future_forecast[future_forecast['ds'] <= last_date + dt.timedelta(days=30)]
avg_next_month = next_month['yhat'].mean()

# 4️⃣ Average of next quarter (90 days)
next_quarter = future_forecast[future_forecast['ds'] <= last_date + dt.timedelta(days=90)]
avg_next_quarter = next_quarter['yhat'].mean()

# 5️⃣ Average of next half year (180 days)
next_half_year = future_forecast[future_forecast['ds'] <= last_date + dt.timedelta(days=180)]
avg_next_half_year = next_half_year['yhat'].mean()

# 6️⃣ Average of next year (365 days)
next_year = future_forecast[future_forecast['ds'] <= last_date + dt.timedelta(days=365)]
avg_next_year = next_year['yhat'].mean()

# Print all results
print("Predicted yhat of Tomorrow:", yhat_tomorrow)
print("Average yhat of Next Week:", avg_next_week)
print("Average yhat of Next Month:", avg_next_month)
print("Average yhat of Next Quarter:", avg_next_quarter)
print("Average yhat of Next Half Year:", avg_next_half_year)
print("Average yhat of Next Year:", avg_next_year)



from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Merge actual and predicted values for the historical period
df_merged = df_prophet.merge(forecast[['ds', 'yhat']], on='ds')

# Actual values
y_true = df_merged['y'].values

# Predicted values
y_pred = df_merged['yhat'].values

# 1️⃣ Mean Absolute Error
mae = mean_absolute_error(y_true, y_pred)

# 2️⃣ Mean Squared Error
mse = mean_squared_error(y_true, y_pred)

# 3️⃣ Root Mean Squared Error
rmse = np.sqrt(mse)

# 4️⃣ Mean Absolute Percentage Error
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Print all metrics
print("Accuracy Metrics:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAPE:", mape, "%")
