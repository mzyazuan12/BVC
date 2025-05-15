import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, LayerNormalization, Activation
from tensorflow.keras.layers import Conv1D, Add, Concatenate, Reshape, Lambda, Multiply
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

# Simplified S4/Mamba-inspired State Space Model
def build_ssm_model(input_shape, state_dim=64, ssm_blocks=4, dropout=0.2):
    """Build a simplified State Space Model inspired by S4/Mamba"""
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Initial projection
    x = Conv1D(filters=state_dim, kernel_size=1)(x)
    
    # SSM blocks
    for _ in range(ssm_blocks):
        # Save residual
        res = x
        
        # State space component (simplified)
        # In a full implementation, this would use specialized SSM kernels
        # Here we approximate with 1D convolutions with different dilation rates
        ssm_out = Conv1D(filters=state_dim, kernel_size=3, padding='same', dilation_rate=1)(x)
        ssm_out = Activation('tanh')(ssm_out)
        
        # Gating mechanism (similar to Mamba's selective scan)
        gate = Conv1D(filters=state_dim, kernel_size=3, padding='same')(x)
        gate = Activation('sigmoid')(gate)
        
        # Apply gate
        ssm_out = Multiply()([ssm_out, gate])
        
        # Skip connection
        x = Add()([res, ssm_out])
        x = LayerNormalization()(x)
        x = Dropout(dropout)(x)
    
    # Output projection
    x = Conv1D(filters=state_dim//2, kernel_size=1)(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    
    # Global pooling and prediction
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# N-BEATS Model (Neural Basis Expansion Analysis for Time Series)
def build_nbeats_model(input_shape, num_stacks=2, num_blocks=3, hidden_units=64, theta_dim=4):
    """Build an N-BEATS model with trend-seasonal decomposition"""
    inputs = Input(shape=input_shape)
    
    # Initial input shape
    seq_length, num_features = input_shape
    
    # Stack outputs for final concatenation
    stack_outputs = []
    
    # Forward pass through stacks and blocks
    for stack_id in range(num_stacks):
        # Initialize backcast and forecast tensors
        backcast = inputs
        forecast = None
        
        # Forward pass through blocks
        for block_id in range(num_blocks):
            # Apply fully connected layers with ReLU activations
            block_input = backcast
            for _ in range(4):  # 4 fully connected layers per block
                block_input = Dense(hidden_units, activation='relu')(block_input)
            
            # Generate backcast and forecast
            theta = Dense(2 * theta_dim)(block_input)
            theta_b, theta_f = tf.split(theta, 2, axis=-1)
            
            # Create basis functions (simplified)
            if stack_id == 0:  # Trend stack
                basis_backcast = tf.range(seq_length, dtype=tf.float32) / seq_length
                basis_forecast = tf.range(1, dtype=tf.float32) / 1  # Single step forecast
            else:  # Seasonal stack
                # Create seasonal basis (simplified)
                basis_backcast = tf.sin(tf.range(seq_length, dtype=tf.float32) * (2 * np.pi / seq_length))
                basis_forecast = tf.sin(tf.range(1, dtype=tf.float32) * (2 * np.pi / 1))
            
            # Expand dimensions for broadcasting
            basis_backcast = tf.expand_dims(tf.expand_dims(basis_backcast, 0), -1)
            basis_forecast = tf.expand_dims(tf.expand_dims(basis_forecast, 0), -1)
            
            # Expand theta dimensions
            theta_b = tf.expand_dims(theta_b, 1)
            theta_f = tf.expand_dims(theta_f, 1)
            
            # Compute backcast and forecast
            backcast_block = tf.matmul(theta_b, basis_backcast)
            backcast_block = tf.squeeze(backcast_block, axis=1)
            
            forecast_block = tf.matmul(theta_f, basis_forecast)
            forecast_block = tf.squeeze(forecast_block, axis=1)
            
            # Update backcast and forecast
            backcast = Lambda(lambda x: x[0] - x[1])([backcast, backcast_block])
            
            if forecast is None:
                forecast = forecast_block
            else:
                forecast = Add()([forecast, forecast_block])
        
        # Add stack forecast to outputs
        stack_outputs.append(forecast)
    
    # Combine stack outputs
    if len(stack_outputs) > 1:
        forecast = Add()(stack_outputs)
    else:
        forecast = stack_outputs[0]
    
    # Final output
    outputs = Dense(1)(forecast)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# TSMixer Model (Time Series MLP Mixer)
def build_tsmixer_model(input_shape, hidden_dim=64, num_layers=4, dropout=0.2):
    """Build a TSMixer model with alternating time and channel mixing"""
    inputs = Input(shape=input_shape)
    x = inputs
    
    seq_length, num_features = input_shape
    
    # Initial projection
    x = Dense(hidden_dim)(x)
    
    # TSMixer blocks
    for _ in range(num_layers):
        # Time mixing (apply MLP across time dimension)
        # First, transpose to shape [batch, features, time]
        time_input = tf.transpose(x, perm=[0, 2, 1])
        time_output = Dense(seq_length)(time_input)
        time_output = Activation('gelu')(time_output)
        time_output = Dense(seq_length)(time_output)
        time_output = Dropout(dropout)(time_output)
        # Transpose back to [batch, time, features]
        time_output = tf.transpose(time_output, perm=[0, 2, 1])
        
        # Residual connection
        x = Add()([x, time_output])
        x = LayerNormalization()(x)
        
        # Channel mixing (apply MLP across feature dimension)
        channel_input = x
        channel_output = Dense(hidden_dim*2)(channel_input)
        channel_output = Activation('gelu')(channel_output)
        channel_output = Dense(hidden_dim)(channel_output)
        channel_output = Dropout(dropout)(channel_output)
        
        # Residual connection
        x = Add()([x, channel_output])
        x = LayerNormalization()(x)
    
    # Global pooling and prediction
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

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
    
    # 2) TRAIN MODELS
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # SSM Model
    print("\nTraining SSM Model...")
    ssm_model = build_ssm_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    ssm_model.fit(
        X_train_seq, y_train_seq,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # N-BEATS Model
    print("\nTraining N-BEATS Model...")
    nbeats_model = build_nbeats_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    nbeats_model.fit(
        X_train_seq, y_train_seq,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # TSMixer Model
    print("\nTraining TSMixer Model...")
    tsmixer_model = build_tsmixer_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    tsmixer_model.fit(
        X_train_seq, y_train_seq,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 3) ROLLING FORECAST EVALUATION
    # For each model, we'll do a rolling forecast on the test set
    def rolling_forecast(model):
        preds = []
        X_history = X_train.copy()
        y_history = y_train.copy()
        
        for i in range(len(test_df)):
            # Create sequence from the most recent data
            X_seq = X_history[-seq_length:].reshape(1, seq_length, X_train.shape[1])
            
            # Make prediction
            pred = model.predict(X_seq, verbose=0)[0]
            pred = scaler_y.inverse_transform([[pred]])[0, 0]
            preds.append(pred)
            
            # Update history with the new observation
            X_new = X_test[i:i+1]
            y_new = scaler_y.transform([[y_test[i]]])[0, 0]
            X_history = np.vstack([X_history, X_new])
            y_history = np.append(y_history, y_new)
        
        return pd.Series(preds, index=test_df.index)
    
    # Generate forecasts
    ssm_preds = rolling_forecast(ssm_model)
    nbeats_preds = rolling_forecast(nbeats_model)
    tsmixer_preds = rolling_forecast(tsmixer_model)
    
    # 4) EVALUATE AND PLOT
    evaluate_forecast("SSM (S4/Mamba-inspired)", y_test, ssm_preds)
    evaluate_forecast("N-BEATS", y_test, nbeats_preds)
    evaluate_forecast("TSMixer", y_test, tsmixer_preds)
    
    # Plot results
    plt.figure(figsize=(14, 8))
    
    plt.plot(train_df['Date'], train_df[target_col], label='Train Actual', color='blue', marker='o')
    plt.plot(test_df['Date'], test_df[target_col], label='Test