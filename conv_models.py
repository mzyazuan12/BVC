# conv_models.py (discontinued program)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, Dropout, Activation, LayerNormalization,
    Add, Multiply, Concatenate, ZeroPadding1D, Lambda, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ──────── helpers ────────
def evaluate_forecast(name, actual, pred):
    mae  = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2   = 1 - ss_res/ss_tot if ss_tot != 0 else float('nan')
    acc  = 100.0 * (1 - mae/actual.mean()) if actual.mean() != 0 else float('nan')
    print(f"\n[{name}] MAE: {mae:.2f}  RMSE: {rmse:.2f}  R²: {r2:.2f}  Acc: {acc:.2f}%")

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)

# ──────── models ────────
def build_tcn_model(input_shape, num_filters=64, kernel_size=3,
                    dilations=[1,2,4,8], dropout=0.2):
    inp = Input(shape=input_shape)
    x = inp
    skips = []
    for d in dilations:
        res = x
        pad = (kernel_size - 1) * d
        x = ZeroPadding1D((pad, 0))(x)
        x = Conv1D(num_filters, kernel_size, dilation_rate=d)(x)
        x = LayerNormalization()(x); x = Activation('relu')(x); x = Dropout(dropout)(x)
        x = Conv1D(num_filters, 1)(x)
        x = LayerNormalization()(x); x = Activation('relu')(x); x = Dropout(dropout)(x)
        if res.shape[-1] != num_filters:
            res = Conv1D(num_filters, 1)(res)
        x = Add()([res, x])
        skips.append(x)
    x = Add()(skips) if len(skips) > 1 else skips[0]
    x = Dense(32, activation='relu')(x)
    out = Dense(1)(x[:, -1, :])
    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

def build_wavenet_model(input_shape, num_filters=32, kernel_size=2,
                        dilations=[1,2,4,8,16], dropout=0.2):
    inp = Input(shape=input_shape)
    x = inp
    skips = []
    for d in dilations:
        res = x
        pad = (kernel_size - 1) * d
        x_pad = ZeroPadding1D((pad, 0))(x)
        f = Conv1D(num_filters, kernel_size, dilation_rate=d)(x_pad)
        g = Conv1D(num_filters, kernel_size, dilation_rate=d)(x_pad)
        x = Multiply()([Activation('tanh')(f), Activation('sigmoid')(g)])
        x = Conv1D(num_filters, 1)(x)
        if res.shape[-1] != num_filters:
            res = Conv1D(num_filters, 1)(res)
        x = Add()([res, x])
        skips.append(Conv1D(num_filters, 1)(x))
    x = Add()(skips) if len(skips) > 1 else skips[0]
    x = Activation('relu')(x)
    x = Conv1D(num_filters,1)(x)
    x = Activation('relu')(x); x = Dropout(dropout)(x)
    x = Conv1D(1,1)(x)
    out = Lambda(lambda t: t[:, -1, :])(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

def build_scinet_model(input_shape, num_filters=32, levels=3, dropout=0.2):
    inp = Input(shape=input_shape)
    x = Conv1D(num_filters, 3, padding='same')(inp)
    x = LayerNormalization()(x); x = Activation('relu')(x)
    for _ in range(levels):
        # split even/odd
        even = Lambda(lambda t: t[:, ::2, :],
                      output_shape=lambda s: (s[0], s[1]//2, s[2]))(x)
        odd  = Lambda(lambda t: t[:, 1::2, :],
                      output_shape=lambda s: (s[0], s[1]//2, s[2]))(x)
        # interact
        e2o = Conv1D(num_filters, 3, padding='same')(even)
        o2e = Conv1D(num_filters, 3, padding='same')(odd)
        even = Add()([even, o2e])
        odd  = Add()([odd, e2o])
        x = Concatenate(axis=1)([even, odd])
        x = LayerNormalization()(x); x = Activation('relu')(x); x = Dropout(dropout)(x)
    # head: pool then dense
    x = Conv1D(num_filters, 1, padding='same')(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    model = Model(inp, outputs)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

# ──────── training pipeline ────────
def main():
    # load & prep
    df = pd.read_csv('d.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    target = 'Gross_Revenue'
    exogs = ['Net_Gas_Price','Corn_Price','CPI','Exchange_Rate_JPY_USD']
    split = int(len(df)*0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    sx, sy = StandardScaler(), StandardScaler()
    X_train = sx.fit_transform(train[exogs])
    y_train = sy.fit_transform(train[[target]]).flatten()
    X_test  = sx.transform(test[exogs])
    y_test  = test[target].values

    seq_len = 24
    X_seq, y_seq = create_sequences(X_train, y_train, seq_len)

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # TCN
    print("\nTraining TCN Model...")
    tcn = build_tcn_model((seq_len, X_seq.shape[2]))
    tcn.fit(
        X_seq, y_seq,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[es],
        verbose=1
    )

    # WaveNet
    print("\nTraining WaveNet Model...")
    wavenet = build_wavenet_model((seq_len, X_seq.shape[2]))
    wavenet.fit(
        X_seq, y_seq,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[es],
        verbose=1
    )

    # SCINet
    print("\nTraining SCINet Model...")
    scinet = build_scinet_model((seq_len, X_seq.shape[2]))
    scinet.fit(
        X_seq, y_seq,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[es],
        verbose=1
    )

    # rolling forecasts
    def rolling_forecast(model):
        preds, hist = [], X_train.copy()
        for i in range(len(test)):
            seg = hist[-seq_len:].reshape(1, seq_len, hist.shape[1])
            scaled = model.predict(seg, verbose=0)   # shape (1,1)
            preds.append(sy.inverse_transform(scaled)[0,0])
            hist = np.vstack([hist, X_test[i:i+1]])
        return np.array(preds)

    tcn_p   = rolling_forecast(tcn)
    wave_p  = rolling_forecast(wavenet)
    sci_p   = rolling_forecast(scinet)

    # evaluate
    evaluate_forecast("TCN",     y_test, tcn_p)
    evaluate_forecast("WaveNet", y_test, wave_p)
    evaluate_forecast("SCINet",  y_test, sci_p)

    # plot
    plt.figure(figsize=(14,8))
    plt.plot(train['Date'], train[target], 'o-', label='Train')
    plt.plot(test ['Date'],  test [target], 'k--o', label='Test')
    plt.plot(test ['Date'],  tcn_p,    'r-x',  label='TCN')
    plt.plot(test ['Date'],  wave_p,   'g-+',  label='WaveNet')
    plt.plot(test ['Date'],  sci_p,    'm-*',  label='SCINet')
    plt.axvline(test['Date'].iloc[0], linestyle='--', color='gray')
    plt.title("Conv Models: Actual vs Predicted")
    plt.xlabel("Date"); plt.ylabel(target); plt.legend(); plt.grid(); plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
