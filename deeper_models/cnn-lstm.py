
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dropout, Dense
)
from tensorflow.keras.optimizers import Adam

# 1) LOAD & FEATURE ENGINEER
# ---------------------------------------------------
DATA_PATH   = '/Users/mac/BFV_model/d.csv'
TARGET_COL  = 'Gross_Revenue'
TEST_RATIO  = 0.2
LOOK_BACK   = 10   # you can tune this

# reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# load + sort
df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# calendar features
df['dow'] = df['Date'].dt.weekday    # 0=Mon,…6=Sun
df['month'] = df['Date'].dt.month    # 1–12
# cyclic encode them
df['sin_dow']   = np.sin(2*np.pi * df['dow']/7)
df['cos_dow']   = np.cos(2*np.pi * df['dow']/7)
df['sin_month'] = np.sin(2*np.pi * (df['month']-1)/12)
df['cos_month'] = np.cos(2*np.pi * (df['month']-1)/12)

# keep only the features we need
features = ['sin_dow','cos_dow','sin_month','cos_month', TARGET_COL]
data = df[features].values
dates = df['Date']

# train/test split
split_idx = int(len(data) * (1 - TEST_RATIO))
train_data = data[:split_idx]
test_data  = data[split_idx:]

# scale all features to [0,1]
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled  = scaler.transform(test_data)


# 2) CREATE SEQUENCES
# ---------------------------------------------------
def create_sequences(dataset, look_back):
    """
    dataset: np.array of shape (n, n_features)
    returns X: (n-look_back, look_back, n_features)
            y: (n-look_back,)
    """
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back, :])          # all features
        y.append(dataset[i+look_back, -1])           # target is last column
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, LOOK_BACK)
X_test,  y_test  = create_sequences(test_scaled,  LOOK_BACK)


# 3) BUILD CNN-LSTM MODEL
# ---------------------------------------------------
n_features = X_train.shape[2]

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu',
           input_shape=(LOOK_BACK, n_features)),
    MaxPooling1D(pool_size=2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)   # linear output
])

model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
model.summary()


# 4) TRAIN
# ---------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    verbose=2
)


# 5) EVALUATE & FORECAST
# ---------------------------------------------------
y_pred_scaled = model.predict(X_test).reshape(-1,1)

# Only inverse-transform the target column
# Build a full-array placeholder to invert properly:
inv_pred_input = np.zeros((len(y_pred_scaled), data.shape[1]))
inv_pred_input[:, -1] = y_pred_scaled[:,0]
y_pred = scaler.inverse_transform(inv_pred_input)[:,-1]

inv_true_input = np.zeros((len(y_test), data.shape[1]))
inv_true_input[:, -1] = y_test
y_true = scaler.inverse_transform(inv_true_input)[:,-1]

# metrics
rmse     = np.sqrt(mean_squared_error(y_true, y_pred))
mae      = mean_absolute_error(y_true, y_pred)
mape     = np.mean(np.abs((y_true - y_pred) / np.where(y_true==0,1,y_true))) * 100
r2       = r2_score(y_true, y_pred)
accuracy = 1 - (mae / np.mean(y_true))

print("\n=== Test Performance ===")
print(f"RMSE     : {rmse:.2f}")
print(f"MAE      : {mae:.2f}")
print(f"MAPE     : {mape:.2f}%")
print(f"R²       : {r2:.2f}")
print(f"Accuracy : {accuracy:.4f}")


# 6) PLOT
# ---------------------------------------------------
plt.figure(figsize=(14,6))

# (a) loss curves
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

# (b) actual vs forecast
plt.subplot(1,2,2)
# reconstruct full actual series
full_scaled = np.vstack([train_scaled, test_scaled])
full_actual = scaler.inverse_transform(full_scaled)[:,-1]
plt.plot(dates, full_actual, label='Actual', marker='o')

# aligned pred dates
num_preds  = len(y_pred)
pred_dates = dates.iloc[split_idx + LOOK_BACK : split_idx + LOOK_BACK + num_preds]
plt.plot(pred_dates, y_pred, label='Forecast', marker='x')

plt.axvline(x=dates.iloc[split_idx], color='gray',
            linestyle='--', label='Train/Test Split')
plt.title('CNN-LSTM Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel(TARGET_COL)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
