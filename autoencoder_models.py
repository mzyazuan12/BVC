import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, LayerNormalization, Activation
from tensorflow.keras.layers import Conv1D, Add, Concatenate, Reshape, Lambda, GRU, LSTM
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
    """Create sequences for time series models"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

# LSTM Autoencoder for Anomaly Detection
def build_lstm_autoencoder(input_shape, latent_dim=16, dropout=0.1):
    """Build an LSTM-based autoencoder for anomaly detection"""
    # Encoder
    inputs = Input(shape=input_shape)
    
    encoded = LSTM(64, return_sequences=True)(inputs)
    encoded = Dropout(dropout)(encoded)
    encoded = LSTM(32, return_sequences=False)(encoded)
    encoded = Dropout(dropout)(encoded)
    encoded = Dense(latent_dim)(encoded)
    
    # Decoder
    decoded = RepeatVector(input_shape[0])(encoded)
    decoded = LSTM(32, return_sequences=True)(decoded)
    decoded = Dropout(dropout)(decoded)
    decoded = LSTM(64, return_sequences=True)(decoded)
    decoded = Dropout(dropout)(decoded)
    decoded = Dense(input_shape[1])(decoded)
    
    # Autoencoder model
    autoencoder = Model(inputs=inputs, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Encoder model for extracting features
    encoder = Model(inputs=inputs, outputs=encoded)
    
    return autoencoder, encoder

# Convolutional Autoencoder
def build_conv_autoencoder(input_shape, latent_dim=16, dropout=0.1):
    """Build a convolutional autoencoder for time series"""
    # Encoder
    inputs = Input(shape=input_shape)
    
    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = LayerNormalization()(x)
    x = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)
    x = Conv1D(filters=8, kernel_size=3, padding='same', activation='relu')(x)
    
    # Global pooling to get latent representation
    encoded = GlobalAveragePooling1D()(x)
    encoded = Dense(latent_dim)(encoded)
    
    # Decoder
    x = Dense(input_shape[0] * 8)(encoded)
    x = Reshape((input_shape[0], 8))(x)
    
    x = Conv1D(filters=8, kernel_size=3, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)
    x = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)
    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)
    
    decoded = Conv1D(filters=input_shape[1], kernel_size=3, padding='same')(x)
    
    # Autoencoder model
    autoencoder = Model(inputs=inputs, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Encoder model for extracting features
    encoder = Model(inputs=inputs, outputs=encoded)
    
    return autoencoder, encoder

# Transformer Autoencoder with Masked Value Pre-training
def build_transformer_autoencoder(input_shape, head_size=32, num_heads=4, ff_dim=128, dropout=0.1):
    """Build a transformer-based autoencoder with masked value pre-training"""
    inputs = Input(shape=input_shape)
    mask_inputs = Input(shape=input_shape)  # Binary mask for masked values
    
    # Add positional encoding
    seq_length, features = input_shape
    pos_encoding = positional_encoding(seq_length, features)
    x = inputs + pos_encoding
    
    # Transformer encoder blocks
    for _ in range(2):
        x = transformer_encoder_block(x, head_size, num_heads, ff_dim, dropout)
    
    # Latent representation
    encoded = GlobalAveragePooling1D()(x)
    
    # Transformer decoder blocks
    x = Dense(features)(encoded)
    x = RepeatVector(seq_length)(x)
    
    for _ in range(2):
        x = transformer_encoder_block(x, head_size, num_heads, ff_dim, dropout)
    
    # Output projection
    decoded = Dense(features)(x)
    
    # Apply mask for loss calculation
    masked_output = Lambda(lambda x: x[0] * x[1])([decoded, mask_inputs])
    masked_inputs = Lambda(lambda x: x[0] * x[1])([inputs, mask_inputs])
    
    # Autoencoder model
    autoencoder = Model(inputs=[inputs, mask_inputs], outputs=masked_output)
    
    # Custom loss that only considers masked values
    def masked_mse(y_true, y_pred):
        # Only calculate loss on masked values
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss=masked_mse)
    
    # Encoder model for extracting features
    encoder = Model(inputs=inputs, outputs=encoded)
    
    return autoencoder, encoder

# Contrastive Predictive Coding (CPC) for Time Series
def build_cpc_model(input_shape, prediction_steps=12, hidden_size=64, dropout=0.1):
    """Build a Contrastive Predictive Coding model for self-supervised learning"""
    # Encoder network
    encoder_inputs = Input(shape=input_shape)
    
    encoder = Conv1D(filters=hidden_size, kernel_size=3, padding='same')(encoder_inputs)
    encoder = LayerNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = Dropout(dropout)(encoder)
    
    encoder = Conv1D(filters=hidden_size, kernel_size=3, padding='same')(encoder)
    encoder = LayerNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = Dropout(dropout)(encoder)
    
    # Context network (GRU)
    context = GRU(hidden_size)(encoder)
    
    # Prediction network (for each future step)
    predictors = [Dense(hidden_size) for _ in range(prediction_steps)]
    
    # Define CPC model
    cpc_model = Model(inputs=encoder_inputs, outputs=context)
    
    # Training model with InfoNCE loss
    def info_nce_loss(context, positive_samples, negative_samples):
        # Implement InfoNCE loss for contrastive learning
        # This is a simplified version
        positive_logits = tf.reduce_sum(context * positive_samples, axis=1)
        negative_logits = tf.reduce_sum(context * negative_samples, axis=1)
        
        return -tf.reduce_mean(positive_logits - tf.math.log(tf.exp(positive_logits) + tf.exp(negative_logits)))
    
    return cpc_model

# Helper functions for transformer models
def positional_encoding(seq_length, d_model):
    """Create positional encodings for transformer models"""
    positions = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_enc = np.zeros((seq_length, d_model))
    pos_enc[:, 0::2] = np.sin(positions * div_term)
    pos_enc[:, 1::2] = np.cos(positions * div_term)
    
    return tf.cast(pos_enc[np.newaxis, ...], dtype=tf.float32)

def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    """Transformer encoder block with multi-head attention"""
    # Multi-head attention
    attention_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    
    # Add & Norm (first residual connection)
    x1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed-forward network
    ffn_output = Dense(ff_dim, activation='relu')(x1)
    ffn_output = Dropout(dropout)(ffn_output)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    
    # Add & Norm (second residual connection)
    return LayerNormalization(epsilon=1e-6)(x1 + ffn_output)

# Custom layers
class RepeatVector(tf.keras.layers.Layer):
    def __init__(self, n, **kwargs):
        super(RepeatVector, self).__init__(**kwargs)
        self.n = n
    
    def call(self, inputs):
        return tf.repeat(tf.expand_dims(inputs, axis=1), repeats=self.n, axis=1)

class GlobalAveragePooling1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalAveragePooling1D, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, key_dim, num_heads, dropout=0.0, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.mha = tf.keras.layers.MultiHeadAttention(
            key_dim=key_dim, num_heads=num_heads, dropout=dropout
        )
    
    def call(self, query, value):
        return self.mha(query=query, value=value)

def main():
    # 1) LOAD AND PREPARE DATA
    df = pd.read_csv('d.csv', parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    target_col = 'Gross_Revenue'
    exog_cols = ['Net_Gas_Price', 'Corn_Price', 'CPI', 'Exchange_Rate_JPY_USD']
    
    # Split data: 80% train, 20% test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(train_df[exog_cols])
    y_train = scaler_y.fit_transform(train_df[[target_col]]).flatten()
    
    X_test = scaler_X.transform(test_df[exog_cols])
    y_test = test_df[target_col].values
    
    # Create sequences for models
    seq_length = 24  # Use 24 months of history
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    
    # 2) TRAIN AUTOENCODER MODELS
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # LSTM Autoencoder
    print("\nTraining LSTM Autoencoder...")
    lstm_autoencoder, lstm_encoder = build_lstm_autoencoder(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    lstm_autoencoder.fit(
        X_train_seq, X_train_seq,  # Autoencoder reconstructs the input
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Convolutional Autoencoder
    print("\nTraining Convolutional Autoencoder...")
    conv_autoencoder, conv_encoder = build_conv_autoencoder(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    conv_autoencoder.fit(
        X_train_seq, X_train_seq,  # Autoencoder reconstructs the input
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 3) ANOMALY DETECTION USING RECONSTRUCTION ERROR
    # Generate reconstructions
    lstm_reconstructions = lstm_autoencoder.predict(X_train_seq)
    conv_reconstructions = conv_autoencoder.predict(X_train_seq)
    
    # Calculate reconstruction errors
    lstm_errors = np.mean(np.square(X_train_seq - lstm_reconstructions), axis=(1, 2))
    conv_errors = np.mean(np.square(X_train_seq - conv_reconstructions), axis=(1, 2))
    
    # Set threshold for anomaly detection (e.g., 95th percentile)
    lstm_threshold = np.percentile(lstm_errors, 95)
    conv_threshold = np.percentile(conv_errors, 95)
    
    # Detect anomalies
    lstm_anomalies = lstm_errors > lstm_threshold
    conv_anomalies = conv_errors > conv_threshold
    
    # 4) FEATURE EXTRACTION FOR DOWNSTREAM TASKS
    # Extract features using trained encoders
    lstm_features = lstm_encoder.predict(X_train_seq)
    conv_features = conv_encoder.predict(X_train_seq)
    
    # 5) FORECASTING USING EXTRACTED FEATURES
    # Train a simple model on the extracted features
    forecast_model_lstm = Sequential([
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(1)
    ])
    forecast_model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    forecast_model_conv = Sequential([
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(1)
    ])
    forecast_model_conv.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Train forecast models
    forecast_model_lstm.fit(lstm_features, y_train_seq, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    forecast_model_conv.fit(conv_features, y_train_seq, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    
    # 6) ROLLING FORECAST EVALUATION
    def rolling_forecast_with_autoencoder(encoder, forecast_model):
        preds = []
        X_history = X_train.copy()
        y_history = y_train.copy()
        
        for i in range(len(test_df)):
            # Create sequence from the most recent data
            X_seq = X_history[-seq_length:].reshape(1, seq_length, X_train.shape[1])
            
            # Extract features using encoder
            features = encoder.predict(X_seq, verbose=0)
            
            # Make prediction using forecast model
            pred = forecast_model.predict(features, verbose=0)[0, 0]
            pred = scaler_y.inverse_transform([[pred]])[0, 0]
            preds.append(pred)
            
            # Update history with the new observation
            X_new = X_test[i:i+1]
            y_new = scaler_y.transform([[y_test[i]]])[0, 0]
            X_history = np.vstack([X_history, X_new])
            y_history = np.append(y_history, y_new)
        
        return pd.Series(preds, index=test_df.index)
    
    # Generate forecasts
    lstm_ae_preds = rolling_forecast_with_autoencoder(lstm_encoder, forecast_model_lstm)
    conv_ae_preds = rolling_forecast_with_autoencoder(conv_encoder, forecast_model_conv)
    
    # 7) EVALUATE AND PLOT
    evaluate_forecast("LSTM Autoencoder", y_test, lstm_ae_preds)
    evaluate_forecast("Conv Autoencoder", y_test, conv_ae_preds)
    
    # Plot forecasting results
    plt.figure(figsize=(14, 8))
    
    plt.plot(train_df['Date'], train_df[target_col], label='Train Actual', color='blue', marker='o')
    plt.plot(test_df['Date'], test_df[target_col], label='Test Actual', color='black', marker='o')
    plt.plot(test_df['Date'], lstm_ae_preds, label='LSTM-AE Prediction', color='red', marker='x')
    plt.plot(test_df['Date'], conv_ae_preds, label='Conv-AE Prediction', color='green', marker='+')
    
    plt.axvline(x=test_df['Date'].iloc[0], color='gray', linestyle='--', label='Train-Test Split')
    plt.title("Autoencoder-based Models: Actual vs. Predicted")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot anomaly detection results
    plt.figure(figsize=(14, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(lstm_errors, label='LSTM Reconstruction Error')
    plt.axhline(y=lstm_threshold, color='r', linestyle='--', label='Threshold')
    plt.title("LSTM Autoencoder Reconstruction Error")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(conv_errors, label='Conv Reconstruction Error')
    plt.axhline(y=conv_threshold, color='r', linestyle='--', label='Threshold')
    plt.title("Convolutional Autoencoder Reconstruction Error")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()