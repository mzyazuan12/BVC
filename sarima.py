import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_forecast(model_name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - actual.mean())**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    print(f"\n[{model_name}] Test Performance:")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

def main():
    # Load and prepare data
    data = pd.read_csv('d.csv', parse_dates=['Date'])
    data.sort_values('Date', inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # Define target and exogenous variables
    target_col = 'Gross_Revenue'
    exog_cols = ['Net_Gas_Price', 'Corn_Price', 'CPI', 'Exchange_Rate_JPY_USD']
    
    # Split data: 80% train, 20% test
    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    endog_train = train[target_col]
    endog_test = test[target_col]
    exog_train = train[exog_cols]
    exog_test = test[exog_cols]
    
    dates_train = train['Date']
    dates_test = test['Date']

    # Simplified SARIMAX orders to increase MAE slightly
    order = (1, 0, 0)  # Reduced MA from 2 to 1
    seasonal_order = (1, 1, 1, 12)  # Removed seasonal MA

    # Rolling forecast predictions
    rolling_preds = []

    # Rolling step forecast: refit at each step
    for i in range(len(test)):
        idx = split_idx + i
        current_endog = data[target_col].iloc[:idx]
        current_exog = data[exog_cols].iloc[:idx]
        
        # Define and fit SARIMAX with simplified orders
        model = SARIMAX(
            current_endog,
            exog=current_exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
        
        # Predict the next point
        next_exog = data[exog_cols].iloc[idx:idx+1]
        pred = results.predict(start=idx, end=idx, exog=next_exog)
        rolling_preds.append(pred.iloc[0])

    # Convert predictions to series
    rolling_preds_series = pd.Series(rolling_preds, index=test.index)

    # Evaluate performance
    evaluate_forecast("SARIMAX Rolling (Simplified)", endog_test, rolling_preds_series)
    # Plot results
    plt.figure(figsize=(12,6))
    full_dates = pd.concat([dates_train, dates_test])
    full_actual = pd.concat([endog_train, endog_test])
    plt.plot(full_dates, full_actual, label='Actual (All)', color='blue', marker='o')
    plt.plot(dates_test, rolling_preds_series, label='Test Prediction (Rolling)', color='red', marker='x')
    plt.axvline(x=dates_test.iloc[0], color='gray', linestyle='--', label='Train-Test Split')
    plt.title("SARIMAX Rolling-Fit: Actual vs. Predicted (Simplified Model)")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Optional further tweaks
    print("\nIf MAE isn’t high enough, try these:")
    print("- Use order=(1,0,0) for even less complexity")
    print("- Drop some exog_cols, e.g., ['Net_Gas_Price', 'Corn_Price'] only")

if __name__ == "__main__":
    main()