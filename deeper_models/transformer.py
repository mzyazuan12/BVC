
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

DATA_PATH   = '/Users/mac/BFV_model/d.csv'
TARGET_COL  = 'Gross_Revenue'
TEST_RATIO  = 0.2
LOOK_BACK   = 20     # sequence length
D_MODEL     = 64     # embedding dim
NUM_HEADS   = 4
FF_DIM      = 128    # feed-forward hidden dim
NUM_LAYERS  = 3      # number of encoder blocks
DROPOUT     = 0.1
BATCH_SIZE  = 32
EPOCHS      = 100
LR          = 1e-3
PATIENCE_ES = 10
PATIENCE_LR = 5

# reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 2) LOAD DATA & FEATURE ENGINEERING
df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# calendar features
df['dow']   = df['Date'].dt.weekday
df['month'] = df['Date'].dt.month
df['sin_dow']   = np.sin(2 * np.pi * df['dow']   / 7)
df['cos_dow']   = np.cos(2 * np.pi * df['dow']   / 7)
df['sin_mo']    = np.sin(2 * np.pi * (df['month']-1) / 12)
df['cos_mo']    = np.cos(2 * np.pi * (df['month']-1) / 12)

# rolling statistics
df['roll7_mean']  = df[TARGET_COL].rolling(window=7,  min_periods=1).mean()
df['roll7_std']   = df[TARGET_COL].rolling(window=7,  min_periods=1).std().fillna(0)
df['roll30_mean'] = df[TARGET_COL].rolling(window=30, min_periods=1).mean()
df['roll30_std']  = df[TARGET_COL].rolling(window=30, min_periods=1).std().fillna(0)

# keep features + target
features = [
    'sin_dow','cos_dow','sin_mo','cos_mo',
    'roll7_mean','roll7_std','roll30_mean','roll30_std',
    TARGET_COL
]
data = df[features].values
dates = df['Date']

# split
split_idx = int(len(data) * (1 - TEST_RATIO))
train_data = data[:split_idx]
test_data  = data[split_idx:]

# scale
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled  = scaler.transform(test_data)

# 3) SEQUENCE CREATION

def create_sequences(arr, look_back):
    X, y = [], []
    for i in range(len(arr) - look_back):
        X.append(arr[i:i+look_back, :-1])  # all but last col
        y.append(arr[i+look_back,  -1])    # last col = target
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, LOOK_BACK)
X_test,  y_test  = create_sequences(test_scaled,  LOOK_BACK)


# 4) POSITIONAL ENCODING

class PositionalEncoding(layers.Layer):
    def __init__(self, seq_len, d_model):
        super().__init__()
        pos = np.arange(seq_len)[:, None]
        i   = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2*(i//2)) / np.float32(d_model))
        angle_rads  = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.cast(angle_rads[None, ...], tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

# 5) TRANSFORMER BLOCK BUILDER

def transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout):
    # Multi-Head Self-Attention
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn = layers.Dropout(dropout)(attn)
    x1   = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    # Feed-Forward
    ff  = layers.Dense(ff_dim, activation='relu')(x1)
    ff  = layers.Dense(d_model)(ff)
    ff  = layers.Dropout(dropout)(ff)
    out = layers.LayerNormalization(epsilon=1e-6)(x1 + ff)
    return out

# 6) BUILD THE MODEL

n_feats = X_train.shape[2]
inp = layers.Input(shape=(LOOK_BACK, n_feats))

# project features to d_model dims
x = layers.Dense(D_MODEL)(inp)
x = PositionalEncoding(LOOK_BACK, D_MODEL)(x)

# stack encoder blocks
for _ in range(NUM_LAYERS):
    x = transformer_encoder_block(x, D_MODEL, NUM_HEADS, FF_DIM, DROPOUT)

# global pooling + head
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(FF_DIM//2, activation='relu')(x)
x = layers.Dropout(DROPOUT)(x)
out = layers.Dense(1)(x)

model = models.Model(inputs=inp, outputs=out)
model.compile(
    optimizer=optimizers.Adam(learning_rate=LR),
    loss='mse'
)
model.summary()

# 7) CALLBACKS: ES, LR REDUCTION, CHECKPOINT

os.makedirs('checkpoints', exist_ok=True)
es = callbacks.EarlyStopping(
    monitor='val_loss', patience=PATIENCE_ES,
    restore_best_weights=True, verbose=1
)
rlr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5,
    patience=PATIENCE_LR, min_lr=1e-6, verbose=1
)
mc = callbacks.ModelCheckpoint(
    'checkpoints/transformer_best.weights.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# 8) TRAIN
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es, rlr, mc],
    verbose=2
)

model.load_weights('checkpoints/transformer_best.weights.h5')

y_pred_scaled = model.predict(X_test).reshape(-1,1)

# inverse transform only the target
inv_pred = np.zeros((len(y_pred_scaled), data.shape[1]))
inv_pred[:, -1] = y_pred_scaled[:,0]
y_pred = scaler.inverse_transform(inv_pred)[:, -1]

inv_true = np.zeros((len(y_test), data.shape[1]))
inv_true[:, -1] = y_test
y_true = scaler.inverse_transform(inv_true)[:, -1]

# metrics
rmse     = np.sqrt(mean_squared_error(y_true, y_pred))
mae      = mean_absolute_error(y_true, y_pred)
mape     = np.mean(np.abs((y_true - y_pred) /
                  np.where(y_true==0,1,y_true))) * 100
r2       = r2_score(y_true, y_pred)
accuracy = 1 - (mae / np.mean(y_true))

print("\n=== Test Set Performance ===")
print(f"RMSE     : {rmse:.2f}")
print(f"MAE      : {mae:.2f}")
print(f"MAPE     : {mape:.2f}%")
print(f"R²       : {r2:.2f}")
print(f"Accuracy : {accuracy:.4f}")

# 10) PLOT RESULTS

plt.figure(figsize=(14,6))

# (a) training history
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

# (b) actual vs aligned forecast
plt.subplot(1,2,2)
full_scaled = np.vstack([train_scaled, test_scaled])
full_actual = scaler.inverse_transform(full_scaled)[:, -1]
plt.plot(dates, full_actual, label='Actual', marker='o')

num_preds  = len(y_pred)
pred_dates = dates.iloc[split_idx : split_idx + num_preds]
plt.plot(pred_dates, y_pred, label='Forecast', marker='x')

plt.axvline(
    x=dates.iloc[split_idx],
    color='gray', linestyle='--', label='Train/Test Split'
)

plt.title('Transformer (Best) Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel(TARGET_COL)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
