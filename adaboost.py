import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit  # Keeping this in case you later need cross-validation
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# 1. Data Loading and Preprocessing
df = pd.read_csv("s.csv", parse_dates=["Date"], index_col="Date")
df = df.sort_index()

numeric_cols = ['Gross_Revenue', 'Exchange_Rate_JPY_USD', 'Net_Gas_Price', 'CPI', 'Corn_Price']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=numeric_cols)

# 2. Feature Engineering
df['trend'] = np.arange(len(df))
df['month'] = df.index.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['Gross_Revenue_lag1'] = df['Gross_Revenue'].shift(1)
df['Gross_Revenue_lag12'] = df['Gross_Revenue'].shift(12)  # yearly lag
df['Gross_Revenue_roll3'] = df['Gross_Revenue'].rolling(window=3).mean()
df = df.dropna()  # Remove rows with NA from shifting/rolling

features = ['trend', 'Exchange_Rate_JPY_USD', 'Net_Gas_Price', 'CPI', 'Corn_Price',
            'month_sin', 'month_cos', 'Gross_Revenue_lag1', 'Gross_Revenue_lag12', 'Gross_Revenue_roll3']
target = 'Gross_Revenue'
X = df[features]
y = df[target]

# 3. Train/Validation/Test Split (80%/10%/10%)
n_total = len(df)
train_size = int(0.8 * n_total)
val_size = int(0.1 * n_total)

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]
X_val = X.iloc[train_size:train_size+val_size]
y_val = y.iloc[train_size:train_size+val_size]
X_test = X.iloc[train_size+val_size:]
y_test = y.iloc[train_size+val_size:]

print(f"Total observations: {n_total}")
print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# 4. Define and Train the Neural Network Model
def build_nn_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Build the model with the number of input features
model = build_nn_model(X_train.shape[1])

# Use EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model using the training and validation sets
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# 5. Predictions on Validation and Test Sets
y_pred_val_nn = model.predict(X_val).flatten()
y_pred_test_nn = model.predict(X_test).flatten()

# 6. Evaluation Metrics
mae_val = mean_absolute_error(y_val, y_pred_val_nn)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val_nn))
r2_val = r2_score(y_val, y_pred_val_nn)
mape_val = np.mean(np.abs((y_val - y_pred_val_nn) / y_val)) * 100

mae_test = mean_absolute_error(y_test, y_pred_test_nn)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test_nn))
r2_test = r2_score(y_test, y_pred_test_nn)
mape_test = np.mean(np.abs((y_test - y_pred_test_nn) / y_test)) * 100

print("\nValidation Metrics (Neural Network):")
print(f"MAE: {mae_val:.2f}")
print(f"RMSE: {rmse_val:.2f}")
print(f"R²: {r2_val:.4f}")
print(f"MAPE: {mape_val:.2f}%")

print("\nTest Metrics (Neural Network):")
print(f"MAE: {mae_test:.2f}")
print(f"RMSE: {rmse_test:.2f}")
print(f"R²: {r2_test:.4f}")
print(f"MAPE: {mape_test:.2f}%")

# 7. Visualization: Forecast Results Plotting Function
def plot_forecast_results(y_train, y_val, y_test, val_preds, test_preds):
    plt.figure(figsize=(16, 8))
    
    # Plot historical data
    plt.plot(y_train.index, y_train, label='Training Data', lw=2)
    plt.plot(y_val.index, y_val, label='Validation Data', lw=2)
    plt.plot(y_test.index, y_test, label='Test Data', lw=2)
    
    # Plot predictions for validation and test sets
    plt.plot(y_val.index, val_preds, '--', label='Validation Predictions', lw=2)
    plt.plot(y_test.index, test_preds, '--', label='Test Predictions', lw=2)
    
    # Mark the start of validation and test periods
    plt.axvline(x=y_val.index[0], color='gray', linestyle=':', label='Validation Start')
    plt.axvline(x=y_test.index[0], color='black', linestyle=':', label='Test Start')
    
    plt.title("Gross Revenue Forecasting Performance with Neural Network", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Gross Revenue", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_forecast_results(y_train, y_val, y_test, y_pred_val_nn, y_pred_test_nn)
