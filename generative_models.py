import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, LayerNormalization, Activation
from tensorflow.keras.layers import Conv1D, Add, Concatenate, Reshape, Lambda, GRU
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

# Diffusion Model for Time Series Forecasting
def build_diffusion_model(input_shape, hidden_size=64, num_steps=50, dropout=0.1):
    """Build a simplified diffusion model for time series forecasting"""
    inputs = Input(shape=input_shape)
    time_step = Input(shape=(1,))
    
    # Encode time step (diffusion step) using sinusoidal encoding
    step_enc = tf.cast(time_step, tf.float32)
    step_enc = tf.expand_dims(step_enc, -1)
    
    # Project input
    x = Conv1D(filters=hidden_size, kernel_size=3, padding='same')(inputs)
    x = LayerNormalization()(x)
    x = Activation('swish')(x)
    
    # Time embedding
    time_emb = Dense(hidden_size, activation='swish')(step_enc)
    time_emb = Dense(hidden_size, activation='swish')(time_emb)
    
    # Reshape time embedding for broadcasting
    time_emb = tf.reshape(time_emb, [-1, 1, hidden_size])
    
    # Add time embedding
    x = x + time_emb
    
    # Backbone network
    skip = x
    x = Conv1D(filters=hidden_size, kernel_size=3, padding='same')(x)
    x = LayerNormalization()(x)
    x = Activation('swish')(x)
    x = Dropout(dropout)(x)
    
    x = Conv1D(filters=hidden_size, kernel_size=3, padding='same')(x)
    x = LayerNormalization()(x)
    x = Activation('swish')(x)
    x = Dropout(dropout)(x)
    
    x = Add()([x, skip])
    
    # Output projection
    x = Conv1D(filters=hidden_size//2, kernel_size=1)(x)
    x = Activation('swish')(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=[inputs, time_step], outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# TimeGrad Model (Normalizing Flow for Time Series)
def build_timegrad_model(input_shape, hidden_size=64, num_flows=2, dropout=0.1):
    """Build a simplified TimeGrad model with normalizing flows"""
    inputs = Input(shape=input_shape)
    
    # Encoder
    x = Conv1D(filters=hidden_size, kernel_size=3, padding='same')(inputs)
    x = LayerNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    
    # Context encoding
    context = GlobalAveragePooling1D()(x)
    context = Dense(hidden_size, activation='relu')(context)
    
    # Simplified flow-based model
    # In a full implementation, this would use proper normalizing flows
    # Here we approximate with a series of transformations
    z = Dense(hidden_size, activation='relu')(context)
    
    for _ in range(num_flows):
        # Affine transformation (simplified flow)
        scale = Dense(1, activation='softplus')(z)
        shift = Dense(1)(z)
        z = z * scale + shift
    
    # Output parameters for Gaussian distribution
    mu = Dense(1)(z)
    sigma = Dense(1, activation='softplus')(z)
    
    # Combine outputs
    outputs = Concatenate()([mu, sigma])
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Custom loss for normalizing flow (negative log likelihood)
    def nll_loss(y_true, y_pred):
        mu, sigma = y_pred[:, 0:1], y_pred[:, 1:2]
        return tf.reduce_mean(0.5 * tf.square((y_true - mu) / sigma) + tf.math.log(sigma))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss=nll_loss)
    return model

# TimeGAN Model (GAN for Time Series)
def build_timegan_model(input_shape, latent_dim=32, hidden_size=64, dropout=0.1):
    """Build a simplified TimeGAN model"""
    # Generator
    def build_generator():
        noise_input = Input(shape=(latent_dim,))
        context_input = Input(shape=input_shape)
        
        # Encode context
        context = Conv1D(filters=hidden_size, kernel_size=3, padding='same')(context_input)
        context = LayerNormalization()(context)
        context = Activation('relu')(context)
        context = GlobalAveragePooling1D()(context)
        
        # Combine noise and context
        x = Concatenate()([noise_input, context])
        
        # Generate sequence
        x = Dense(hidden_size, activation='relu')(x)
        x = Dense(hidden_size, activation='relu')(x)
        x = Dense(1)(x)
        
        return Model([noise_input, context_input], x)
    
    # Discriminator
    def build_discriminator():
        real_input = Input(shape=(1,))
        context_input = Input(shape=input_shape)
        
        # Encode context
        context = Conv1D(filters=hidden_size, kernel_size=3, padding='same')(context_input)
        context = LayerNormalization()(context)
        context = Activation('relu')(context)
        context = GlobalAveragePooling1D()(context)
        
        # Combine real data and context
        x = Concatenate()([real_input, context])
        
        # Discriminate
        x = Dense(hidden_size, activation='relu')(x)
        x = Dropout(dropout)(x)
        x = Dense(hidden_size, activation='relu')(x)
        x = Dropout(dropout)(x)
        x = Dense(1, activation='sigmoid')(x)
        
        return Model([real_input, context_input], x)
    
    # Build and compile the discriminator
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))
    
    # Build the generator
    generator = build_generator()
    
    # For the combined model, we only train the generator
    discriminator.trainable = False
    
    # The combined model
    noise_input = Input(shape=(latent_dim,))
    context_input = Input(shape=input_shape)
    generated_output = generator([noise_input, context_input])
    validity = discriminator([generated_output, context_input])
    
    combined = Model([noise_input, context_input], validity)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))
    
    return generator, discriminator, combined

# Global Average Pooling Layer
class GlobalAveragePooling1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalAveragePooling1D, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)

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
    
    # TimeGrad Model
    print("\nTraining TimeGrad Model...")
    timegrad_model = build_timegrad_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    timegrad_model.fit(
        X_train_seq, y_train_seq,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Diffusion Model (simplified training)
    print("\nTraining Diffusion Model...")
    diffusion_model = build_diffusion_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    
    # Simplified diffusion training (just train on clean data with step=0)
    time_steps = np.zeros((len(X_train_seq), 1))
    diffusion_model.fit(
        [X_train_seq, time_steps], y_train_seq,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # TimeGAN Model
    print("\nTraining TimeGAN Model...")
    latent_dim = 32
    generator, discriminator, combined = build_timegan_model(
        input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
        latent_dim=latent_dim
    )
    
    # Simplified GAN training (just a few epochs for demonstration)
    batch_size = 32
    epochs = 10
    
    # Train the GAN
    for epoch in range(epochs):
        # Select a random batch of samples
        idx = np.random.randint(0, X_train_seq.shape[0], batch_size)
        real_seqs = X_train_seq[idx]
        real_targets = y_train_seq[idx].reshape(-1, 1)
        
        # Generate noise
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        
        # Generate predictions
        gen_targets = generator.predict([noise, real_seqs])
        
        # Train the discriminator
        d_loss_real = discriminator.train_on_batch([real_targets, real_seqs], np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch([gen_targets, real_seqs], np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the generator
        g_loss = combined.train_on_batch([noise, real_seqs], np.ones((batch_size, 1)))
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
    
    # 3) ROLLING FORECAST EVALUATION
    # For each model, we'll do a rolling forecast on the test set
    def rolling_forecast_timegrad(model):
        preds = []
        X_history = X_train.copy()
        y_history = y_train.copy()
        
        for i in range(len(test_df)):
            # Create sequence from the most recent data
            X_seq = X_history[-seq_length:].reshape(1, seq_length, X_train.shape[1])
            
            # Make prediction
            pred_params = model.predict(X_seq, verbose=0)[0]
            mu, sigma = pred_params[0], pred_params[1]
            pred = scaler_y.inverse_transform([[mu]])[0, 0]
            preds.append(pred)
            
            # Update history with the new observation
            X_new = X_test[i:i+1]
            y_new = scaler_y.transform([[y_test[i]]])[0, 0]
            X_history = np.vstack([X_history, X_new])
            y_history = np.append(y_history, y_new)
        
        return pd.Series(preds, index=test_df.index)
    
    def rolling_forecast_diffusion(model):
        preds = []
        X_history = X_train.copy()
        y_history = y_train.copy()
        
        for i in range(len(test_df)):
            # Create sequence from the most recent data
            X_seq = X_history[-seq_length:].reshape(1, seq_length, X_train.shape[1])
            time_step = np.zeros((1, 1))  # Use step 0 for clean prediction
            
            # Make prediction
            pred = model.predict([X_seq, time_step], verbose=0)[0]
            pred = scaler_y.inverse_transform([[pred]])[0, 0]
            preds.append(pred)
            
            # Update history with the new observation
            X_new = X_test[i:i+1]
            y_new = scaler_y.transform([[y_test[i]]])[0, 0]
            X_history = np.vstack([X_history, X_new])
            y_history = np.append(y_history, y_new)
        
        return pd.Series(preds, index=test_df.index)
    
    def rolling_forecast_timegan(generator, latent_dim):
        preds = []
        X_history = X_train.copy()
        y_history = y_train.copy()
        
        for i in range(len(test_df)):
            # Create sequence from the most recent data
            X_seq = X_history[-seq_length:].reshape(1, seq_length, X_train.shape[1])
            
            # Generate noise
            noise = np.random.normal(0, 1, (1, latent_dim))
            
            # Make prediction
            pred = generator.predict([noise, X_seq], verbose=0)[0, 0]
            pred = scaler_y.inverse_transform([[pred]])[0, 0]
            preds.append(pred)
            
            # Update history with the new observation
            X_new = X_test[i:i+1]
            y_new = scaler_y.transform([[y_test[i]]])[0, 0]
            X_history = np.vstack([X_history, X_new])
            y_history = np.append(y_history, y_new)
        
        return pd.Series(preds, index=test_df.index)
    
    # Generate forecasts
    timegrad_preds = rolling_forecast_timegrad(timegrad_model)
    diffusion_preds = rolling_forecast_diffusion(diffusion_model)
    timegan_preds = rolling_forecast_timegan(generator, latent_dim)
    
    # 4) EVALUATE AND PLOT
    evaluate_forecast("TimeGrad", y_test, timegrad_preds)
    evaluate_forecast("Diffusion", y_test, diffusion_preds)
    evaluate_forecast("TimeGAN", y_test, timegan_preds)
    
    # Plot results
    plt.figure(figsize=(14, 8))
    
    plt.plot(train_df['Date'], train_df[target_col], label='Train Actual', color='blue', marker='o')
    plt.plot(test_df['Date'], test_df[target_col], label='Test Actual', color='black', marker='o')
    plt.plot(test_df['Date'], timegrad_preds, label='TimeGrad Prediction', color='red', marker='x')
    plt.plot(test_df['Date'], diffusion_preds, label='Diffusion Prediction', color='green', marker='+')
    plt.plot(test_df['Date'], timegan_preds, label='TimeGAN Prediction', color='purple', marker='*')
    
    plt.axvline(x=test_df['Date'].iloc[0], color='gray', linestyle='--', label='Train-Test Split')
    plt.title("Generative Models Comparison: Actual vs. Predicted")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()