
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
from xgboost import XGBRegressor
import optuna

DATA_PATH = '/Users/mac/BFV_model/d.csv'
TARGET_COL = 'Gross_Revenue'
TEST_RATIO = 0.2  # 80% train, 20% test

data = pd.read_csv(DATA_PATH, parse_dates=['Date'])
data.sort_values('Date', inplace=True)
data.reset_index(drop=True, inplace=True)

values    = data[[TARGET_COL]].values        # shape (n,1)
split_idx = int(len(values) * (1 - TEST_RATIO))
train_vals, test_vals = values[:split_idx], values[split_idx:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_vals)
test_scaled  = scaler.transform(test_vals)

def create_dataset(series: np.ndarray, look_back: int):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i : i + look_back, 0])
        y.append(series[i + look_back, 0])
    return np.array(X), np.array(y)

def objective(trial):
  
    look_back        = trial.suggest_int("look_back", 1, 10)
    n_estimators     = trial.suggest_int("n_estimators", 50, 500)
    max_depth        = trial.suggest_int("max_depth", 3, 12)
    learning_rate    = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    subsample        = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    gamma            = trial.suggest_float("gamma", 0.0, 5.0)
    reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 5.0)
    reg_lambda       = trial.suggest_float("reg_lambda", 0.0, 5.0)

    X_train, y_train = create_dataset(train_scaled, look_back)
    X_valid, y_valid = create_dataset(test_scaled,  look_back)

    # Check if datasets are empty, which can happen if look_back is too large for test set
    if X_valid.shape[0] == 0 or y_valid.shape[0] == 0:
        print(f"Warning: Validation set empty for look_back={look_back}. Returning high RMSE.")
        return float('inf') # Return a large value to penalize this trial

    model = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        # use_label_encoder=False, # Deprecated in newer versions, safe to remove or keep as False if needed
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        verbosity=0, # Suppress XGBoost messages during Optuna trials
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=20, # Use the older parameter
        verbose=False # Suppress verbose output during fitting in trials
    )

    preds = model.predict(X_valid)
    return np.sqrt(mean_squared_error(y_valid, preds))

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    # Increase n_trials for better hyperparameter search
    study.optimize(objective, n_trials=50) # Increased trials to 50

    print("\n=== Best Trial ===")
    trial = study.best_trial
    print(f"Validation RMSE: {trial.value:.4f}") # Clarify this is validation RMSE
    print("Best Parameters:")
    for k, v in trial.params.items():
        print(f"  {k}: {v}")

    best_params = trial.params.copy()
    look_back   = best_params.pop("look_back")

    # Create datasets using the best look_back
    X_train, y_train = create_dataset(train_scaled, look_back)
    X_test,  y_test  = create_dataset(test_scaled,  look_back) # This is the actual test set

    # Check if test set is valid
    if X_test.shape[0] == 0 or y_test.shape[0] == 0:
         print(f"\nError: Test set has size 0 with look_back={look_back}. Cannot train final model.")
         print(f"Test data length: {len(test_scaled)}, Required length: {look_back + 1}")
         exit() # Stop execution if test set is invalid

    final_model = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        # use_label_encoder=False, # Deprecated
        **best_params # Unpack all other best hyperparameters
        # n_estimators is already optimized by Optuna, use that value
    )

    print("\nTraining final model...")
    final_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)], # Evaluate on the test set
        early_stopping_rounds=20,   # Use the older parameter
        verbose=True # Show fitting progress for the final model
    )

    # Predict using the best iteration found during the final fit
    y_pred_scaled = final_model.predict(X_test).reshape(-1,1)
    y_pred        = scaler.inverse_transform(y_pred_scaled)
    y_true        = scaler.inverse_transform(y_test.reshape(-1,1))

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    # Handle potential division by zero in MAPE
    y_true_safe = np.where(y_true == 0, 1e-6, y_true) # Replace 0 with a small number
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    r2   = r2_score(y_true, y_pred)

    print("\n=== Final Test Metrics ===")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2:.2f}")

    dates       = data['Date'].values
    # Adjust indices for plotting based on look_back
    train_dates = dates[look_back:split_idx] # Dates for actual training targets
    test_dates  = dates[split_idx + look_back:] # Dates for actual test targets/predictions

    # Ensure plot arrays match lengths
    if len(test_dates) != len(y_pred):
         print("\nWarning: Length mismatch between test dates and predictions. Plot might be misaligned.")
         # Adjust test_dates to match prediction length if necessary
         test_dates = dates[split_idx + look_back : split_idx + look_back + len(y_pred)]


    full_actual = data[[TARGET_COL]].values # Use original unscaled data for plotting actuals

    plt.figure(figsize=(15,7)) # Wider figure
    # Plot all actual data
    plt.plot(dates, full_actual.flatten(), label='Actual Gross Revenue', marker='.', linestyle='-', color='blue', alpha=0.7)
    # Plot predictions on the test set range
    plt.plot(test_dates, y_pred.flatten(), label='XGBoost Forecast', marker='x', linestyle='--', color='red')

    # Add split line
    plt.axvline(x=dates[split_idx], color='gray', linestyle=':', linewidth=2, label=f'Train/Test Split ({1-TEST_RATIO:.0%}/{TEST_RATIO:.0%})')

    plt.title("XGBoost Forecast vs Actual Gross Revenue", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(TARGET_COL, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout() # Adjust layout to prevent overlap
    plt.show()