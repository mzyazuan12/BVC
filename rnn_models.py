import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
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
    """Create sequences for RNN models"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

def build_gru_model(input_shape, units=64, dropout=0.2):
    """Build a GRU model"""
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        GRU(units//2),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def build_deepar_model(input_shape, units=64, dropout=0.2):
    """Build a DeepAR-like model (Amazon's probabilistic RNN)"""
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        GRU(units//2, return_sequences=True),
        Dropout(dropout),
        GRU(units//4),
        Dense(2)  # Output mean and std for probabilistic forecasting
    ])
    
    # Custom loss for probabilistic forecasting (negative log likelihood)
    def nll_loss(y_true, y_pred):
        mean, std = y_pred[:, 0:1], tf.math.softplus(y_pred[:, 1:2])
        gaussian = tf.random.normal(shape=tf.shape(mean))
        z = (y_true - mean) / (std + 1e-6)
        return tf.reduce_mean(0.5 * tf.square(z) + tf.math.log(std + 1e-6))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss=nll_loss)
    return model

def build_deepstate_model(input_shape, units=64, dropout=0.2):
    """Build a DeepState-like model (State-space RNN)"""
    # State-space models combine classical state-space filtering with RNNs
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        GRU(units//2, return_sequences=True),
        Dropout(dropout),
        GRU(units//4),
        Dense(3)  # Output state transition parameters
    ])
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
    
    # Create sequences for RNN
    seq_length = 12  # Use 12 months of history
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    
    # 2) TRAIN MODELS
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # GRU Model
    print("\nTraining GRU Model...")
    gru_model = build_gru_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    gru_model.fit(
        X_train_seq, y_train_seq,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # DeepAR Model
    print("\nTraining DeepAR Model...")
    deepar_model = build_deepar_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    deepar_model.fit(
        X_train_seq, y_train_seq.reshape(-1, 1),
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # DeepState Model
    print("\nTraining DeepState Model...")
    deepstate_model = build_deepstate_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    deepstate_model.fit(
        X_train_seq, np.column_stack([y_train_seq, y_train_seq, y_train_seq]),  # Placeholder for state params
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 3) ROLLING FORECAST EVALUATION
    # For each model, we'll do a rolling forecast on the test set
    def rolling_forecast(model, model_type='standard'):
        preds = []
        X_history = X_train.copy()
        y_history = y_train.copy()
        
        for i in range(len(test_df)):
            # Create sequence from the most recent data
            X_seq = X_history[-seq_length:].reshape(1, seq_length, X_train.shape[1])
            
            # Make prediction
            if model_type == 'standard':
                pred = model.predict(X_seq, verbose=0)[0, 0]
                pred = scaler_y.inverse_transform([[pred]])[0, 0]
            elif model_type == 'deepar':
                pred_params = model.predict(X_seq, verbose=0)[0]
                mean, std = pred_params[0], tf.math.softplus(pred_params[1])
                pred = scaler_y.inverse_transform([[mean]])[0, 0]
            elif model_type == 'deepstate':
                # Simplified state-space prediction
                pred_params = model.predict(X_seq, verbose=0)[0]
                pred = scaler_y.inverse_transform([[pred_params[0]]])[0, 0]
            
            preds.append(pred)
            
            # Update history with the new observation
            X_new = X_test[i:i+1]
            y_new = scaler_y.transform([[y_test[i]]])[0, 0]
            X_history = np.vstack([X_history, X_new])
            y_history = np.append(y_history, y_new)
        
        return pd.Series(preds, index=test_df.index)
    
    # Generate forecasts
    gru_preds = rolling_forecast(gru_model, 'standard')
    deepar_preds = rolling_forecast(deepar_model, 'deepar')
    deepstate_preds = rolling_forecast(deepstate_model, 'deepstate')
    
    # 4) EVALUATE AND PLOT
    evaluate_forecast("GRU", y_test, gru_preds)
    evaluate_forecast("DeepAR", y_test, deepar_preds)
    evaluate_forecast("DeepState", y_test, deepstate_preds)
    
    # Plot results
    plt.figure(figsize=(14, 8))
    
    plt.plot(train_df['Date'], train_df[target_col], label='Train Actual', color='blue', marker='o')
    plt.plot(test_df['Date'], test_df[target_col], label='Test Actual', color='black', marker='o')
    plt.plot(test_df['Date'], gru_preds, label='GRU Prediction', color='red', marker='x')
    plt.plot(test_df['Date'], deepar_preds, label='DeepAR Prediction', color='green', marker='+')
    plt.plot(test_df['Date'], deepstate_preds, label='DeepState Prediction', color='purple', marker='*')
    
    plt.axvline(x=test_df['Date'].iloc[0], color='gray', linestyle='--', label='Train-Test Split')
    plt.title("RNN Models Comparison: Actual vs. Predicted")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()