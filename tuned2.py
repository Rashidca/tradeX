import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Step 1: Load and prepare the dataset
# ------------------------
df = pd.read_csv("/Users/afrahanas/Downloads/project3/AAPL_all_data2.csv", header=0, skiprows=[1,2])
df.rename(columns={'Price':'Date'}, inplace=True)
df_prophet = df[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)

# ------------------------
# Step 2: 80/20 Train-Test Split
# ------------------------
split_index = int(len(df_prophet) * 0.8)
train_df = df_prophet.iloc[:split_index]
test_df = df_prophet.iloc[split_index:]

print(f"Training data: {train_df['ds'].min()} to {train_df['ds'].max()}")
print(f"Testing data: {test_df['ds'].min()} to {test_df['ds'].max()}")

# ------------------------
# Step 3: Fit Prophet with fine-tuned hyperparameters
# ------------------------
model = Prophet(
    daily_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.05,  # controls trend flexibility
    seasonality_prior_scale=10.0,  # controls seasonality strength
    seasonality_mode='additive'    # try 'multiplicative' if trends grow with magnitude
)
model.fit(train_df)

# ------------------------
# Step 4: Forecast on test set
# ------------------------
future_test = test_df[['ds']]
forecast_test = model.predict(future_test)

# ------------------------
# Step 5: Evaluate accuracy on test set
# ------------------------
results_df = pd.merge(test_df, forecast_test[['ds','yhat','yhat_lower','yhat_upper']], on='ds')

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

# ------------------------
# Step 7: Forecast next 1 year beyond dataset
# ------------------------
future_full = model.make_future_dataframe(periods=365)
forecast_full = model.predict(future_full)

plt.figure(figsize=(14,7))
plt.plot(df_prophet['ds'], df_prophet['y'], label='Historical Price', color='blue')
plt.plot(forecast_full['ds'], forecast_full['yhat'], label='Forecasted Price', color='red', linestyle='--')
plt.fill_between(forecast_full['ds'], forecast_full['yhat_lower'], forecast_full['yhat_upper'], color='red', alpha=0.2)
plt.title('Apple Stock Price Forecast (Next 1 Year)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
