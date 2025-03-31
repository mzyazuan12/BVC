import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_arima_model(y_train, y_test, order):
    try:
        model = ARIMA(y_train, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(y_test))
        rmse = np.sqrt(mean_squared_error(y_test, forecast))
        return rmse, forecast, model_fit
    except Exception as e:
        return np.inf, None, None

# Load and sort data
data = pd.read_csv('d.csv', parse_dates=['Date'])
data.sort_values('Date', inplace=True)
data.reset_index(drop=True, inplace=True)

# Define target variable and split into train and test sets
target_col = 'Gross_Revenue'
split_idx = int(len(data) * 0.8)
train_data = data.iloc[:split_idx]
test_data  = data.iloc[split_idx:]

y_train = train_data[target_col].values
y_test  = test_data[target_col].values

# Define candidate orders for ARIMA (adjust these ranges as needed)
p_values = range(0, 4)   # try 0 to 3
d_values = range(0, 3)   # try 0 to 2
q_values = range(0, 4)   # try 0 to 3

best_rmse = np.inf
best_order = None
best_model = None
improvement = True
iteration = 0
max_iter = 10  # safety stop

while improvement and iteration < max_iter:
    improvement = False
    iteration += 1
    print(f"\nIteration {iteration}:")
    # Loop over all candidate orders
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                rmse, forecast, model_fit = evaluate_arima_model(y_train, y_test, order)
                if rmse < best_rmse:
                    print(f"  New best order {order} with RMSE {rmse:.4f}")
                    best_rmse = rmse
                    best_order = order
                    best_model = model_fit
                    improvement = True
    if not improvement:
        print("No further improvement found in this iteration.")
        break

print(f"\nBest ARIMA order found: {best_order} with RMSE {best_rmse:.4f}")

# Forecast on test set with the best model
forecast = best_model.forecast(steps=len(y_test))

# Compute Evaluation Metrics
mae = mean_absolute_error(y_test, forecast)
mse = mean_squared_error(y_test, forecast)
rmse = np.sqrt(mse)  # This should match best_rmse
r2 = 1 - np.sum((y_test - forecast)**2) / np.sum((y_test - np.mean(y_test))**2)
mape = np.mean(np.abs((y_test - forecast) / y_test)) * 100
accuracy = 1 - (mae / np.mean(y_test))

print("\nEvaluation Metrics on Test Set:")
print(f"  Mean Absolute Error (MAE): {mae:.4f}")
print(f"  Mean Squared Error (MSE): {mse:.4f}")
print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"  R^2 Score: {r2:.4f}")
print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"  Accuracy (1 - MAE/mean): {accuracy:.4f}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], np.concatenate([y_train, y_test]), label='Actual', marker='o')
plt.axvline(x=test_data['Date'].iloc[0], color='gray', linestyle='--', label='Train-Test Split')
plt.plot(test_data['Date'], forecast, label='Forecast', marker='x')
plt.title("Iteratively Tuned ARIMA Model Forecast")
plt.xlabel("Date")
plt.ylabel(target_col)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
