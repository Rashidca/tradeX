import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import itertools
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt


# ----------------------------
# Simple Prophet helper
# ----------------------------
def run_prophet_forecast(df):
    """
    Runs a Prophet forecast on the given stock DataFrame.
    Expected columns: 'Date' and 'Close'
    Returns forecast DataFrame or None if error occurs.
    """
    try:
        if df is None or df.empty:
            print("❌ Error: DataFrame is empty or None.")
            return None

        if 'Date' not in df.columns or 'Close' not in df.columns:
            print("❌ Error: DataFrame must contain 'Date' and 'Close' columns.")
            print("Columns found:", df.columns.tolist())
            return None

        df = df[['Date', 'Close']].copy()
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna()

        if df.empty:
            print("❌ Error: No valid rows after cleaning data.")
            return None

        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        print("✅ Prophet forecast generated successfully.")
        return forecast

    except Exception as e:
        print("❌ Error running Prophet forecast:", e)
        return None

def forecast_aapl_cv(df, periods=365, save_csv_path=None):
    """
    Forecast stock prices using Prophet with 80/20 split and CV-based hyperparameter tuning.
    Accepts a DataFrame directly. Optionally saves the DataFrame to CSV.

    Returns:
        model: trained Prophet model
        forecast_full: forecast including future
        test_results: merged dataframe of test actuals and predictions
        final_metrics: dict of MAE, RMSE, MAPE
    """

    # ------------------------
    # Step 0: Optional save to CSV
    # ------------------------
    if save_csv_path is not None:
        df.to_csv(save_csv_path, index=True)
        print(f"✅ Saved stock data to {save_csv_path}")

    # ------------------------
    # Step 1: Preprocess for Prophet
    # ------------------------
    if 'Date' in df.columns and 'Close' in df.columns:
        df_prophet = df[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
    else:
        raise ValueError("DataFrame must contain 'Date' and 'Close' columns")

    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)

    # ------------------------
    # Step 2: 80/20 Train-Test split
    # ------------------------
    split_index = int(len(df_prophet) * 0.8)
    train_df = df_prophet.iloc[:split_index]
    test_df = df_prophet.iloc[split_index:]

    # ------------------------
    # Step 3: Hyperparameter tuning (may take time)
    # ------------------------
    grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1],
        'seasonality_prior_scale': [1.0, 5.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }

    all_params = [dict(zip(grid.keys(), v)) for v in itertools.product(*grid.values())]
    rmses = []

    for params in all_params:
        try:
            model_cv = Prophet(daily_seasonality=True, yearly_seasonality=True, **params).fit(train_df)
            df_cv = cross_validation(model_cv, initial='730 days', period='180 days', horizon='365 days', parallel="processes")
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])
        except Exception as e:
            print(f"⚠️ Skipped params {params} due to error: {e}")
            rmses.append(np.inf)

    # Best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    best_params = tuning_results.loc[tuning_results['rmse'].idxmin()].to_dict()
    best_params.pop('rmse')

    print("✅ Best Parameters found:", best_params)

    # ------------------------
    # Step 4: Train final model
    # ------------------------
    model = Prophet(daily_seasonality=True, yearly_seasonality=True, **best_params)
    model.fit(train_df)

    # Forecast test set
    forecast_test = model.predict(test_df[['ds']])
    results_df = pd.merge(test_df, forecast_test[['ds','yhat','yhat_lower','yhat_upper']], on='ds', how='left')

    # Debug print
    print("🔍 results_df shape:", results_df.shape)
    print("🔍 Columns:", results_df.columns)
    print("🔍 Sample rows:\n", results_df.head())

    # ------------------------
    # Step 5: Accuracy metrics
    # ------------------------
    if not results_df.empty:
        y_true = results_df['y'].squeeze()
        y_pred = results_df['yhat'].squeeze()

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)
        final_metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    else:
        final_metrics = {'MAE': None, 'RMSE': None, 'MAPE': None}
        print("⚠️ Warning: Empty results_df, skipping metrics calculation.")

    # ------------------------
    # Step 6: Forecast future
    # ------------------------
    future_full = model.make_future_dataframe(periods=periods)
    forecast_full = model.predict(future_full)

    return model, forecast_full, results_df, final_metrics


def calculate_future_averages(forecast_full, last_date):
    """
    Calculate average forecasted prices for various future time periods.
    """
    periods_dict = {
        'next_week': 7,
        'next_month': 30,
        'next_quarter': 90,
        'next_half_year': 180,
        'next_year': 365
    }
    averages = {}
    for label, days in periods_dict.items():
        subset = forecast_full[forecast_full['ds'] <= last_date + dt.timedelta(days=days)]
        averages[label] = subset['yhat'].mean()
    return averages


def plot_forecast(forecast_full):
    """
    Plot the full forecast.
    """
    plt.figure(figsize=(10,5))
    plt.plot(forecast_full['ds'], forecast_full['yhat'], label='Forecasted Price', color='red', linestyle='--')
    plt.fill_between(forecast_full['ds'], forecast_full['yhat_lower'], forecast_full['yhat_upper'], color='red', alpha=0.2)
    plt.title('Stock Price Forecast (Next 1 Year)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
