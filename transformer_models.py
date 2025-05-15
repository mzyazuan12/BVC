import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_forecast(model_name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    acc = 100.0 * (1 - mae / actual.mean()) if actual.mean() != 0 else float('nan')
    
    print(f"\n[{model_name}] Test Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")
    print(f"Accuracy (1 - MAE/mean * 100): {acc:.2f}%")

def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

def positional_encoding(seq_length, d_model):
    positions = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_enc = np.zeros((seq_length, d_model))
    pos_enc[:, 0::2] = np.sin(positions * div_term)
    pos_enc[:, 1::2] = np.cos(positions * div_term)
    return tf.cast(pos_enc[np.newaxis, ...], dtype=tf.float32)

def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    x = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    ffn_output = Dense(ff_dim, activation='relu')(x)
    ffn_output = Dropout(dropout)(ffn_output)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    return LayerNormalization(epsilon=1e-6)(x + ffn_output)

def build_ts_transformer_model(input_shape, head_size=64, num_heads=4, ff_dim=256, num_blocks=4,
                               mlp_units=[128], dropout=0.1):
    inputs = Input(shape=input_shape)
    seq_len, features = input_shape
    x = inputs + positional_encoding(seq_len, features)

    for _ in range(num_blocks):
        x = transformer_encoder_block(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D()(x)

    for units in mlp_units:
        x = Dense(units, activation='relu')(x)
        x = Dropout(dropout)(x)

    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

def build_informer_model(input_shape, head_size=64, num_heads=4, ff_dim=256,
                         num_encoder_layers=2, num_decoder_layers=1, dropout=0.1):
    inputs = Input(shape=input_shape)
    seq_len, features = input_shape
    x = inputs + positional_encoding(seq_len, features)

    for _ in range(num_encoder_layers):
        x = transformer_encoder_block(x, head_size, num_heads, ff_dim, dropout)

    for _ in range(num_decoder_layers):
        x = transformer_encoder_block(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

def build_tft_model(input_shape, hidden_size=64, num_heads=4, dropout=0.1):
    inputs = Input(shape=input_shape)
    seq_len, features = input_shape

    x = Dense(hidden_size, activation='elu')(inputs)
    x = x + positional_encoding(seq_len, hidden_size)

    skip = x
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dense(hidden_size, activation='elu')(x)
    x = Dense(hidden_size, activation='elu')(x)
    gate = Dense(hidden_size, activation='sigmoid')(x)
    x = x * gate + skip

    x = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_size // num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)

    skip = x
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dense(hidden_size * 2, activation='elu')(x)
    x = Dense(hidden_size)(x)
    x = x + skip

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

def main():
    df = pd.read_csv('d.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

    target_col = 'Gross_Revenue'
    exog_cols = ['Net_Gas_Price', 'Corn_Price', 'CPI', 'Exchange_Rate_JPY_USD']

    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    scaler_X, scaler_y = StandardScaler(), StandardScaler()

    X_train = scaler_X.fit_transform(train_df[exog_cols])
    y_train = scaler_y.fit_transform(train_df[[target_col]]).flatten()

    X_test = scaler_X.transform(test_df[exog_cols])
    y_test = test_df[target_col].values

    seq_length = 24
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print('\nTraining TimeSeries Transformer …')
    ts_model = build_ts_transformer_model((X_train_seq.shape[1], X_train_seq.shape[2]))
    ts_model.fit(X_train_seq, y_train_seq, epochs=100, batch_size=32,
                 validation_split=0.2, callbacks=[early_stop], verbose=1)

    print('\nTraining Informer …')
    informer = build_informer_model((X_train_seq.shape[1], X_train_seq.shape[2]))
    informer.fit(X_train_seq, y_train_seq, epochs=100, batch_size=32,
                 validation_split=0.2, callbacks=[early_stop], verbose=1)

    print('\nTraining TFT …')
    tft = build_tft_model((X_train_seq.shape[1], X_train_seq.shape[2]))
    tft.fit(X_train_seq, y_train_seq, epochs=100, batch_size=32,
            validation_split=0.2, callbacks=[early_stop], verbose=1)

    def rolling_forecast(model):
        preds = []
        X_hist = X_train.copy()

        for i in range(len(test_df)):
            X_seq = X_hist[-seq_length:].reshape(1, seq_length, X_train.shape[1])

            pred_scaled = model.predict(X_seq, verbose=0)
            pred = scaler_y.inverse_transform(pred_scaled)[0, 0]
            preds.append(pred)

            X_new = X_test[i:i + 1]
            X_hist = np.vstack([X_hist, X_new])

        return pd.Series(preds, index=test_df.index)

    ts_preds = rolling_forecast(ts_model)
    informer_preds = rolling_forecast(informer)
    tft_preds = rolling_forecast(tft)

    evaluate_forecast('TimeSeries Transformer', y_test, ts_preds)
    evaluate_forecast('Informer', y_test, informer_preds)
    evaluate_forecast('TFT', y_test, tft_preds)

    plt.figure(figsize=(14, 8))
    plt.plot(train_df['Date'], train_df[target_col], label='Train Actual', marker='o')
    plt.plot(test_df['Date'], test_df[target_col], label='Test Actual', color='black', marker='o')
    plt.plot(test_df['Date'], ts_preds, label='Transformer Prediction', color='red', marker='x')
    plt.plot(test_df['Date'], informer_preds, label='Informer Prediction', color='green', marker='+')
    plt.plot(test_df['Date'], tft_preds, label='TFT Prediction', color='purple', marker='*')
    plt.axvline(x=test_df['Date'].iloc[0], color='gray', linestyle='--', label='Train-Test Split')
    plt.title('Transformer-based Models: Actual vs. Predicted')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
