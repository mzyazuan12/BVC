
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


DATA_PATH  = '/Users/mac/BFV_model/d.csv'
TARGET_COL = 'Gross_Revenue'
TEST_RATIO = 0.2  # 80% train, 20% test

# load & sort
data = pd.read_csv(DATA_PATH, parse_dates=['Date'])
data.sort_values('Date', inplace=True)
data.reset_index(drop=True, inplace=True)

# extract values and split
values    = data[[TARGET_COL]].values    # shape (n,1)
split_idx = int(len(values) * (1 - TEST_RATIO))
train_vals, test_vals = values[:split_idx], values[split_idx:]

# scale to [0,1]
scaler       = MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(train_vals)
test_scaled  = scaler.transform(test_vals)



def create_dataset(series: np.ndarray, look_back: int):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i : i + look_back, 0])
        y.append(series[i + look_back, 0])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # hyperparameters
    look_back      = 5       # number of past days to use
    hidden_units1  = 64
    hidden_units2  = 32
    dropout_rate   = 0.2
    learning_rate  = 0.001
    batch_size     = 32
    epochs         = 50

    # prepare train/test datasets
    X_train, y_train = create_dataset(train_scaled, look_back)
    X_test,  y_test  = create_dataset(test_scaled,  look_back)

    # DEFINE MODEL
    model = Sequential([
        Dense(hidden_units1, activation='relu', input_shape=(look_back,)),
        Dropout(dropout_rate),
        Dense(hidden_units2, activation='relu'),
        Dense(1)  # linear output
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    # TRAIN
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    #  FORECAST & INVERSE-SCALE
    y_pred_scaled = model.predict(X_test).reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test.reshape(-1,1))

    # METRICS
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true==0,1,y_true))) * 100
    r2   = r2_score(y_true, y_pred)
    accuracy = 1 - (mae / np.mean(y_true))

    print("\n=== Test Set Performance ===")
    print(f"RMSE     : {rmse:.2f}")
    print(f"MAE      : {mae:.2f}")
    print(f"MAPE     : {mape:.2f}%")
    print(f"R²       : {r2:.2f}")
    print(f"Accuracy : {accuracy:.4f}")  

    # PLOT LOSS & FORECAST VS ACTUAL
    plt.figure(figsize=(14,6))

    # loss curves
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    # forecast vs actual
    plt.subplot(1,2,2)
    dates = data['Date'].values
    # align dates
    test_dates = dates[split_idx + look_back:]
    full_scaled = np.vstack([train_scaled, test_scaled])
    full_actual = scaler.inverse_transform(full_scaled)

    plt.plot(dates, full_actual.flatten(), label='Actual', marker='o')
    plt.plot(test_dates, y_pred.flatten(), label='Forecast', marker='x')
    plt.axvline(x=dates[split_idx], color='gray', linestyle='--', label='Split')
    plt.title('ANN Forecast vs Actual')
    plt.xlabel('Date')
    plt.ylabel(TARGET_COL)
    plt.legend()
    plt.tight_layout()
    plt.show()
