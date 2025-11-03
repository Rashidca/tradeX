import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import datetime as dt

# ----------------------------
# 1️⃣ Load and prepare dataset
# ----------------------------
df = pd.read_csv("/Users/afrahanas/Downloads/project3/AAPL_all_data.csv", header=0, skiprows=[1, 2])
df.rename(columns={'Price': 'Date'}, inplace=True)
df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

# ----------------------------
# 2️⃣ Split data 80% train, 20% test
# ----------------------------
train_size = int(len(df_prophet) * 0.8)
train_df = df_prophet.iloc[:train_size]
test_df = df_prophet.iloc[train_size:]

# ----------------------------
# 3️⃣ Train Prophet model on training set
# ----------------------------
model = Prophet()
model.fit(train_df)

# ----------------------------
# 4️⃣ Make future dataframe for the length of test set
# ----------------------------
future = model.make_future_dataframe(periods=len(test_df))
forecast = model.predict(future)

# ----------------------------
# Extract forecast for the test period
# ----------------------------
test_forecast = forecast.iloc[-len(test_df):]  # align with test set
y_pred = test_forecast['yhat'].values
y_true = test_df['y'].values

# Compute accuracy metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("Accuracy Metrics on Test Set:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAPE:", mape, "%")


# ----------------------------
# 7️⃣ Optional: plot forecast vs actual for test period
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(test_df['ds'], y_true, label='Actual')
plt.plot(test_forecast['ds'], y_pred, label='Forecast')
plt.title('Apple Stock Price Forecast vs Actual (Test Set)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
