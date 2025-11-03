import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Step 1: Load the dataset
# ------------------------
df = pd.read_csv("/Users/afrahanas/Downloads/project3/AAPL_all_data2.csv", header=0, skiprows=[1,2])
df.rename(columns={'Price':'Date'}, inplace=True)
df_prophet = df[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
df_prophet = df_prophet.sort_values('ds')

# ------------------------
# Step 2: Split 80% train, 20% test
# ------------------------
split_index = int(len(df_prophet) * 0.8)
train_df = df_prophet.iloc[:split_index]
test_df = df_prophet.iloc[split_index:]

print(f"Training data: {train_df['ds'].min()} to {train_df['ds'].max()}")
print(f"Testing data: {test_df['ds'].min()} to {test_df['ds'].max()}")

# ------------------------
# Step 3: Fit Prophet on training data
# ------------------------
model = Prophet(daily_seasonality=True, yearly_seasonality=True)
model.fit(train_df)

# ------------------------
# Step 4: Forecast on test data
# ------------------------
future = test_df[['ds']]  # only the dates in test set
forecast = model.predict(future)

# ------------------------
# Step 5: Calculate accuracy metrics
# ------------------------
results_df = pd.merge(test_df, forecast[['ds','yhat','yhat_lower','yhat_upper']], on='ds')

mae = mean_absolute_error(results_df['y'], results_df['yhat'])
rmse = np.sqrt(mean_squared_error(results_df['y'], results_df['yhat']))
mape = mean_absolute_percentage_error(results_df['y'], results_df['yhat'])

print("\n--- Accuracy Metrics on Test Set ---")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2%}")

# ------------------------
# Step 6: Plot actual vs predicted
# ------------------------
plt.figure(figsize=(14,7))
plt.plot(results_df['ds'], results_df['y'], label='Actual Price', color='blue')
plt.plot(results_df['ds'], results_df['yhat'], label='Predicted Price', color='red', linestyle='--')
plt.fill_between(results_df['ds'], results_df['yhat_lower'], results_df['yhat_upper'], color='red', alpha=0.2)
plt.title('Apple Stock Price Forecast (80/20 split)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
