import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import itertools

# -----------------------------
# Main guard for macOS multiprocessing safety
# -----------------------------
if __name__ == "__main__":

    # 1️⃣ Load and prepare data
    df = pd.read_csv("/Users/afrahanas/Downloads/project3/AAPL_all_data.csv", 
                     header=0, skiprows=[1, 2])
    df.rename(columns={'Price': 'Date'}, inplace=True)
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    # 2️⃣ Split into training (80%) and testing (20%)
    split_index = int(len(df_prophet) * 0.8)
    train_df = df_prophet.iloc[:split_index]
    test_df = df_prophet.iloc[split_index:]

    print(f"Training data rows: {len(train_df)}, Test data rows: {len(test_df)}\n")

    # 3️⃣ Simple hyperparameter tuning (grid search)
    grid = {
        'changepoint_prior_scale': [0.05, 0.1],
        'seasonality_mode': ['additive', 'multiplicative']
    }

    all_params = [dict(zip(grid.keys(), v)) for v in itertools.product(*grid.values())]
    best_rmse = float('inf')
    best_params = None

    for params in all_params:
        m = Prophet(**params).fit(train_df)
        forecast = m.predict(train_df)
        rmse = np.sqrt(mean_squared_error(train_df['y'], forecast['yhat']))
        print(f"Tested params {params}, RMSE={rmse:.2f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    print("\n--- Best Hyperparameters ---")
    print(best_params)

    # 4️⃣ Fit final model with best parameters
    final_model = Prophet(**best_params)
    final_model.fit(train_df)

    # 5️⃣ Forecast on test set
    future_test = test_df[['ds']]
    forecast_test = final_model.predict(future_test)

    # Merge actual and predicted
    results_df = pd.merge(test_df, forecast_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

    # 6️⃣ Accuracy metrics
    mae = mean_absolute_error(results_df['y'], results_df['yhat'])
    rmse = np.sqrt(mean_squared_error(results_df['y'], results_df['yhat']))
    mape = mean_absolute_percentage_error(results_df['y'], results_df['yhat'])

    print("\n--- Accuracy Metrics on Test Set ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2%}")

    # 7️⃣ Plot actual vs predicted
    plt.figure(figsize=(14, 7))
    plt.plot(results_df['ds'], results_df['y'], label='Actual Price', color='blue')
    plt.plot(results_df['ds'], results_df['yhat'], label='Predicted Price', color='red', linestyle='--')
    plt.fill_between(results_df['ds'], results_df['yhat_lower'], results_df['yhat_upper'], color='red', alpha=0.2)
    plt.title('Apple Stock Price Forecast vs Actuals (Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
