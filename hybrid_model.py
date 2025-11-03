import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
def flatten_dataframe_columns(df):
    """
    Resets the DataFrame index and flattens the MultiIndex columns
    into single, descriptive string names.

    Example:
    - ('Adj Close', 'AAPL') -> 'Price_AAPL'
    - ('Volume', 'AAPL')    -> 'Volume_AAPL'
    - 'Date' (from index)   -> 'Date'
    """

    # Reset the index, bringing index levels (e.g., 'Date') in as columns
    df = df.reset_index()

    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            # This is a MultiIndex column (e.g., ('Adj Close', 'AAPL'))

            # Extract parts, handling potential empty strings in the tuple
            part1 = str(col[0]) if col[0] else ''
            part2 = str(col[1]) if len(col) > 1 and col[1] else ''

            if 'date' in part1.lower():
                new_cols.append('Date')
            elif 'close' in part1.lower():
                # Name it 'Price' or 'Price_TICKER'
                new_cols.append(f"Price_{part2}" if part2 else "Price")
            else:
                # Join parts like 'Volume_AAPL' or just 'Volume'
                new_cols.append(f"{part1}_{part2}" if part2 else part1)
        else:
            # This is a single-string column (e.g., 'Date' from reset_index)
            col_str = str(col)
            if 'date' in col_str.lower():
                new_cols.append('Date')
            elif col_str.lower() == 'index':
                 # Often the default name from reset_index if the index had no name
                 # You might want to rename this or drop it
                 new_cols.append('Date') # Or pass
            else:
                new_cols.append(col_str)

    # Assign the new, clean column names
    df.columns = new_cols

    # Handle potential duplicate 'Date' columns if 'Date' was both
    # in the index and a column tuple.
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    return df

def smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
    Formula: (100% / n) * Σ( |y_pred - y_true| / ((|y_true| + |y_pred|)/2) )
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_pred - y_true) / np.where(denominator == 0, 1, denominator)
    return np.mean(diff) * 100


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = 1e-8
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)

# 
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = 1e-8
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)

def forecast_aapl_hybrid_fixed(df, periods=365, n_lags=5, test_frac=0.2, xgb_params=None):
    """
    Hybrid Prophet + XGBoost (fixed):
      - removes duplicate dates
      - splits BEFORE fitting scalers (no leakage)
      - builds lag features from actual scaled y
      - trains XGB on residuals (y_scaled - yhat_prophet)
    Returns:
      model_prophet, model_xgb, forecast_full_prophet, test_df_with_preds, metrics
    """

    if xgb_params is None:
        xgb_params = dict(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)

    # ----- Detect columns -----
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    close_col = next((c for c in df.columns if 'close' in c.lower() or 'price' in c.lower()), None)
    if not date_col or not close_col:
        raise ValueError("Data must contain 'Date' and 'Close' (or 'Price') columns.")

    df_prophet = df[[date_col, close_col]].rename(columns={date_col: 'ds', close_col: 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)

    # remove duplicate dates (keep last)
    df_prophet = df_prophet.drop_duplicates(subset='ds', keep='last').reset_index(drop=True)

    # ----- Train/test split (on chronological order) -----
    split_idx = int(len(df_prophet) * (1 - test_frac))
    df_train = df_prophet.iloc[:split_idx].copy().reset_index(drop=True)
    df_test = df_prophet.iloc[split_idx:].copy().reset_index(drop=True)

    # ----- Fit scaler(s) on TRAIN only -----
    scaler_y = MinMaxScaler()  # can use StandardScaler if preferred
    df_train['y_scaled'] = scaler_y.fit_transform(df_train[['y']])
    df_test['y_scaled'] = scaler_y.transform(df_test[['y']])

    # ----- Fit Prophet on train's scaled y -----
    model_prophet = Prophet(daily_seasonality=True, yearly_seasonality=True, stan_backend='CMDSTANPY')
    model_prophet.fit(df_train[['ds', 'y_scaled']].rename(columns={'y_scaled': 'y'}))

    # ----- Get Prophet forecasts for train and test (on their ds) -----
    fc_train = model_prophet.predict(df_train[['ds']])
    df_train = df_train.merge(fc_train[['ds', 'yhat']], on='ds').rename(columns={'yhat': 'yhat_prophet'})

    fc_test = model_prophet.predict(df_test[['ds']])
    df_test = df_test.merge(fc_test[['ds', 'yhat']], on='ds').rename(columns={'yhat': 'yhat_prophet'})

    # ----- Residuals on TRAIN (this is the target for XGB) -----
    df_train['residual'] = df_train['y_scaled'] - df_train['yhat_prophet']

    # ----- Create lag features using actual scaled y (important) -----
    # Build lags on the combined series to have continuous lags, then re-split carefully
    df_all = pd.concat([df_train[['ds', 'y_scaled', 'yhat_prophet']], df_test[['ds', 'y_scaled', 'yhat_prophet']]],
                      ignore_index=True).reset_index(drop=True)

    for lag in range(1, n_lags + 1):
        df_all[f'lag_{lag}'] = df_all['y_scaled'].shift(lag)

    # drop rows with NaN lags (these belong to the earliest rows)
    df_all = df_all.dropna().reset_index(drop=True)

    # Now re-derive train/test masks by ds values
    train_ds_max = df_train['ds'].max()
    train_mask = df_all['ds'] <= train_ds_max
    test_mask = df_all['ds'] > train_ds_max

    df_xgb_train = df_all[train_mask].copy().reset_index(drop=True)
    df_xgb_test = df_all[test_mask].copy().reset_index(drop=True)

    # Ensure residual column exists in df_xgb_train by merging from df_train
    df_xgb_train = df_xgb_train.merge(df_train[['ds', 'residual']], on='ds', how='left')

    # Drop any rows where residual is null (shouldn't happen but safe)
    df_xgb_train = df_xgb_train.dropna(subset=['residual']).reset_index(drop=True)

    # Prepare X and y for XGB
    feature_cols = [f'lag_{i}' for i in range(1, n_lags + 1)]
    X_train = df_xgb_train[feature_cols].values
    y_res_train = df_xgb_train['residual'].values

    X_test = df_xgb_test[feature_cols].values
    # Note: we won't have residuals for df_xgb_test (that's what we want to predict)

    # Optionally scale features (fit scaler on X_train only)
    feat_scaler = StandardScaler()
    X_train_scaled = feat_scaler.fit_transform(X_train)
    X_test_scaled = feat_scaler.transform(X_test)

    # ----- Train XGBoost on residuals -----
    model_xgb = XGBRegressor(**xgb_params)
    model_xgb.fit(X_train_scaled, y_res_train)

    # ----- Predict residuals on test portion -----
    xgb_res_pred = model_xgb.predict(X_test_scaled)

    # ----- Compose hybrid predictions (still in scaled y space) -----
    # For the df_xgb_test rows, we have yhat_prophet (scaled), so add predicted residuals
    df_xgb_test['yhat_prophet'] = df_xgb_test['yhat_prophet'].values  # scaled prophet predictions
    df_xgb_test['yhat_hybrid_scaled'] = df_xgb_test['yhat_prophet'].values + xgb_res_pred

    # ----- Bring predictions back to original scale -----
    # Prepare arrays for inverse transform
    prophet_only_scaled = df_xgb_test['yhat_prophet'].values.reshape(-1, 1)
    hybrid_scaled = df_xgb_test['yhat_hybrid_scaled'].values.reshape(-1, 1)
    y_true_scaled = df_xgb_test['y_scaled'].values.reshape(-1, 1)

    yhat_prophet_orig = scaler_y.inverse_transform(prophet_only_scaled).ravel()
    yhat_hybrid_orig = scaler_y.inverse_transform(hybrid_scaled).ravel()
    y_true_orig = scaler_y.inverse_transform(y_true_scaled).ravel()

    # ----- Build final test dataframe for evaluation -----
    out = pd.DataFrame({
        'ds': df_xgb_test['ds'].values,
        'y_true': y_true_orig,
        'yhat_prophet': yhat_prophet_orig,
        'yhat_hybrid': yhat_hybrid_orig
    })

    # ----- Metrics -----
    metrics = {
        'Prophet': {
            'MAE': mean_absolute_error(out['y_true'], out['yhat_prophet']),
            'RMSE': np.sqrt(mean_squared_error(out['y_true'], out['yhat_prophet'])),
            'MAPE': mean_absolute_percentage_error(out['y_true'], out['yhat_prophet']),
            'SMAPE': smape(out['y_true'], out['yhat_prophet'])
        },
        'Hybrid': {
            'MAE': mean_absolute_error(out['y_true'], out['yhat_hybrid']),
            'RMSE': np.sqrt(mean_squared_error(out['y_true'], out['yhat_hybrid'])),
            'MAPE': mean_absolute_percentage_error(out['y_true'], out['yhat_hybrid']),
            'SMAPE': smape(out['y_true'], out['yhat_hybrid'])
        },
        'Diff (Hybrid - Prophet)': {
            'MAE_diff': None,
            'RMSE_diff': None,
            'MAPE_diff': None
        }
    }

    metrics['Diff (Hybrid - Prophet)']['MAE_diff'] = metrics['Hybrid']['MAE'] - metrics['Prophet']['MAE']
    metrics['Diff (Hybrid - Prophet)']['RMSE_diff'] = metrics['Hybrid']['RMSE'] - metrics['Prophet']['RMSE']
    metrics['Diff (Hybrid - Prophet)']['MAPE_diff'] = metrics['Hybrid']['MAPE'] - metrics['Prophet']['MAPE']

    # ----- Return -----
    # forecast_full_prophet: Prophet's forecast for full timeline (train+test+future) in scaled y space
    future_full = model_prophet.make_future_dataframe(periods=periods)
    forecast_full = model_prophet.predict(future_full)


    # -----------------------------------------------
    # NEW: Hybrid future forecasting using XGBoost
    # -----------------------------------------------
    # We'll extend lag features using the last known scaled y values
    df_all_for_future = pd.concat([
        df_all[['ds', 'y_scaled', 'yhat_prophet']],
        forecast_full[['ds', 'yhat']].rename(columns={'yhat': 'yhat_prophet_future'})
    ], ignore_index=True)

    # Replace missing yhat_prophet with forecasted values
    df_all_for_future['yhat_prophet'] = df_all_for_future['yhat_prophet'].combine_first(df_all_for_future['yhat_prophet_future'])

    # Start with known scaled y values
    known_scaled_y = df_all['y_scaled'].tolist()
    hybrid_preds_scaled = []
    forecast_future_prophet = forecast_full.tail(periods).reset_index(drop=True)


    for i in range(periods):
        # Take last n_lags values (whether real or predicted)
        if len(known_scaled_y) < n_lags:
            break  # safety
        lags = known_scaled_y[-n_lags:]
        X_future = np.array(lags).reshape(1, -1)
        X_future_scaled = feat_scaler.transform(X_future)

        # Predict residual correction
        res_future = model_xgb.predict(X_future_scaled)[0]

        # Prophet scaled forecast for this step
        yhat_prophet_scaled = forecast_future_prophet.loc[i, 'yhat']


        # Hybrid scaled forecast
        yhat_hybrid_scaled = yhat_prophet_scaled + res_future

        # Append to known series for next step’s lag generation
        known_scaled_y.append(yhat_hybrid_scaled)
        hybrid_preds_scaled.append(yhat_hybrid_scaled)

    # Inverse-transform to original scale
    hybrid_preds_scaled = np.array(hybrid_preds_scaled).reshape(-1, 1)
    hybrid_preds_orig = scaler_y.inverse_transform(hybrid_preds_scaled).ravel()

    # Attach hybrid predictions to future dataframe
    forecast_future = forecast_full.tail(periods).copy()
    forecast_future['yhat_hybrid'] = hybrid_preds_orig

    # Return everything
    return model_prophet, model_xgb, forecast_full, out, metrics, forecast_future

def summarize_future_prices(forecast_df):
    """
    Takes the hybrid forecast DataFrame (from forecast_future_dataframe)
    and calculates next day price and average forecasts for
    week, month, quarter, half-year, and year.
    """
    # Safety check
    if 'yhat_hybrid' not in forecast_df.columns:
        raise ValueError("Expected column 'yhat_hybrid' not found in forecast_df")

    # Compute different period summaries
    results = {
        'Next Day Price': float(round(forecast_df.iloc[0]['yhat_hybrid'], 2)),
        'Next Week Avg': float(round(forecast_df.head(7)['yhat_hybrid'].mean(), 2)),
        'Next Month Avg': float(round(forecast_df.head(30)['yhat_hybrid'].mean(), 2)),
        'Next Quarter Avg': float(round(forecast_df.head(90)['yhat_hybrid'].mean(), 2)),
        'Next Half-Year Avg': float(round(forecast_df.head(180)['yhat_hybrid'].mean(), 2)),
        'Next Year Avg': float(round(forecast_df.head(365)['yhat_hybrid'].mean(), 2))
    }

    return results
