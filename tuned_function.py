import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import itertools
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

def forecast_aapl_cv(csv_path, periods=365):
    """
    Forecast stock prices using Prophet with CV-based hyperparameter tuning.
    Accepts a CSV file path with 'Date' and 'Close' columns.
    
    Returns (aligned with UI usage):
        model: trained Prophet model
        results_df: merged dataframe of test actuals and predictions
        forecast_full: forecast including future
        final_metrics: dict of MAE, RMSE, MAPE
    """

    # ------------------------
    # Step 1: Read and preprocess data
    # ------------------------
    df = pd.read_csv(csv_path)

    # Try to detect 'Date' and 'Close' columns dynamically
    cols = [col.lower() for col in df.columns]
    date_col = None
    close_col = None
    for c in df.columns:
        if 'date' in c.lower():
            date_col = c
        if 'price' in c.lower():
            close_col = c

    if not date_col or not close_col:
        raise ValueError("CSV must contain 'Date' and 'Close' columns (case-insensitive).")

    df_prophet = df[[date_col, close_col]].rename(columns={date_col: 'ds', close_col: 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)

    # ------------------------
    # Step 2: 80/20 Train-Test split
    # ------------------------
    split_index = int(len(df_prophet) * 0.8)
    train_df = df_prophet.iloc[:split_index]
    test_df = df_prophet.iloc[split_index:]

    # ------------------------
    # Step 3: Hyperparameter tuning
    # ------------------------
    grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1],
        'seasonality_prior_scale': [1.0, 5.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }

    all_params = [dict(zip(grid.keys(), v)) for v in itertools.product(*grid.values())]
    rmses = []

    for params in all_params:
        
        model_cv = Prophet(
    daily_seasonality=True,
    yearly_seasonality=True,
    stan_backend='CMDSTANPY',
    **params
)

        model_cv.fit(train_df)
        try:
            df_cv = cross_validation(model_cv, initial='730 days', period='180 days', horizon='365 days', parallel="processes")
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])
        except Exception:
            rmses.append(np.inf)

    # Best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    best_params = tuning_results.loc[tuning_results['rmse'].idxmin()].to_dict()
    best_params.pop('rmse')

    # ------------------------
    # Step 4: Train final model
    # ------------------------
    model = Prophet(
    daily_seasonality=True,
    yearly_seasonality=True,
    stan_backend='CMDSTANPY',
    **best_params
)

    model.fit(train_df)

    # Forecast test set
    forecast_test = model.predict(test_df[['ds']])
    results_df = pd.merge(test_df, forecast_test[['ds','yhat','yhat_lower','yhat_upper']], on='ds')

    # ------------------------
    # Step 5: Accuracy metrics
    # ------------------------
    mae = mean_absolute_error(results_df['y'], results_df['yhat'])
    rmse = np.sqrt(mean_squared_error(results_df['y'], results_df['yhat']))
    mape = mean_absolute_percentage_error(results_df['y'], results_df['yhat'])
    final_metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    # ------------------------
    # Step 6: Forecast future
    # ------------------------
    future_full = model.make_future_dataframe(periods=periods)
    forecast_full = model.predict(future_full)

    return model, results_df, forecast_full, final_metrics


def calculate_future_averages(forecast_full):
    """
    Calculates mean forecast values for specific future periods.
    """
    last_date = forecast_full['ds'].max()
    periods_dict = {
        'next_week': 7,
        'next_month': 30,
        'next_quarter': 90,
        'next_half_year': 180,
        'next_year': 365
    }
    averages = {}
    for label, days in periods_dict.items():
        subset = forecast_full[forecast_full['ds'] <= last_date - dt.timedelta(days=periods_dict['next_year']) + dt.timedelta(days=days)]
        averages[label] = subset['yhat'].mean()
    return averages




def plot_forecast(forecast_full):
    """
    Simple forecast visualization (Streamlit-safe version).
    Returns a Matplotlib figure instead of calling plt.show().
    """
    # ✅ CHANGE 1: Use fig, ax = plt.subplots() instead of plt.figure()
    fig, ax = plt.subplots(figsize=(10, 5))

    # ✅ CHANGE 2: Plot on ax (no global plt)
    ax.plot(forecast_full['ds'], forecast_full['yhat'], label='Forecasted Price', color='red', linestyle='--')
    ax.fill_between(forecast_full['ds'], forecast_full['yhat_lower'], forecast_full['yhat_upper'],
                    color='red', alpha=0.2)

    # ✅ CHANGE 3: Set labels and formatting using ax
    ax.set_title('Stock Price Forecast (Next 1 Year)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    fig.tight_layout()

    # ✅ CHANGE 4: Return fig (instead of plt.show())
    return fig

