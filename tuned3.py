import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import itertools
import numpy as np
import matplotlib.pyplot as plt

def forecast_cv(file_path, periods=365):
    """
    Forecast Apple stock prices using Prophet with 80/20 split and CV-based hyperparameter tuning.

    Returns:
        model: trained Prophet model
        forecast_full: forecast including future
        test_results: merged dataframe of test actuals and predictions
        final_metrics: dict of MAE, RMSE, MAPE
    """

    # ------------------------
    # Step 1: Load and preprocess
    # ------------------------
    df = pd.read_csv(file_path, header=0, skiprows=[1,2])
    df.rename(columns={'Price':'Date'}, inplace=True)
    df_prophet = df[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)

    # ------------------------
    # Step 2: 80/20 Train-Test split
    # ------------------------
    split_index = int(len(df_prophet) * 0.8)
    train_df = df_prophet.iloc[:split_index]
    test_df = df_prophet.iloc[split_index:]

    print(f"Training data: {train_df['ds'].min()} to {train_df['ds'].max()}")
    print(f"Testing data: {test_df['ds'].min()} to {test_df['ds'].max()}")

    # ------------------------
    # Step 3: Hyperparameter tuning using cross-validation
    # ------------------------
    grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1],
        'seasonality_prior_scale': [1.0, 5.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }

    all_params = [dict(zip(grid.keys(), v)) for v in itertools.product(*grid.values())]
    rmses = []

    print(f"Starting hyperparameter tuning with {len(all_params)} combinations...")

    for params in all_params:
        model_cv = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            **params
        ).fit(train_df)
        try:
            df_cv = cross_validation(model_cv, initial='730 days', period='180 days', horizon='365 days', parallel="processes")
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])
        except Exception as e:
            print(f"Skipping params {params} due to error: {e}")
            rmses.append(np.inf)

    # Select best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    best_params = tuning_results.loc[tuning_results['rmse'].idxmin()].to_dict()
    best_params.pop('rmse')

    print("\nBest Hyperparameters Found:")
    print(best_params)

    # ------------------------
    # Step 4: Train final model
    # ------------------------
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        **best_params
    )
    model.fit(train_df)

    # Forecast test set
    future_test = test_df[['ds']]
    forecast_test = model.predict(future_test)

    results_df = pd.merge(test_df, forecast_test[['ds','yhat','yhat_lower','yhat_upper']], on='ds')

    # ------------------------
    # Step 5: Compute final accuracy metrics
    # ------------------------
    mae = mean_absolute_error(results_df['y'], results_df['yhat'])
    rmse = np.sqrt(mean_squared_error(results_df['y'], results_df['yhat']))
    mape = mean_absolute_percentage_error(results_df['y'], results_df['yhat'])

    final_metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    print("\n--- Final Accuracy Metrics on Test Set ---")
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
    plt.title('Apple Stock Price Forecast (80/20 split, CV-tuned)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # ------------------------
    # Step 7: Forecast next 'periods' days
    # ------------------------
    future_full = model.make_future_dataframe(periods=periods)
    forecast_full = model.predict(future_full)

    plt.figure(figsize=(14,7))
    plt.plot(df_prophet['ds'], df_prophet['y'], label='Historical Price', color='blue')
    plt.plot(forecast_full['ds'], forecast_full['yhat'], label='Forecasted Price', color='red', linestyle='--')
    plt.fill_between(forecast_full['ds'], forecast_full['yhat_lower'], forecast_full['yhat_upper'], color='red', alpha=0.2)
    plt.title(f'Apple Stock Price Forecast (Next {periods} Days)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return model, forecast_full, results_df, final_metrics

# Example usage:
# model, forecast_full, test_results, final_metrics = forecast_aapl_cv("/Users/afrahanas/Downloads/project3/AAPL_all_data2.csv")
forecast_cv("/Users/afrahanas/Downloads/project3/AAPL_all_data2.csv")