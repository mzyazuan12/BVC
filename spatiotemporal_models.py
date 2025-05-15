import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
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

# Create a simple adjacency matrix for demonstration
def create_demo_adjacency_matrix(num_nodes):
    """Create a simple adjacency matrix for demonstration"""
    # For demonstration, we'll create a simple chain graph
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes-1):
        adj_matrix[i, i+1] = 1
        adj_matrix[i+1, i] = 1
    return adj_matrix

# Graph Convolutional Layer
class GraphConvLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(GraphConvLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        # Input shape: [batch_size, num_nodes, features]
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='bias'
        )
        self.built = True
    
    def call(self, inputs, adj_matrix):
        # inputs: [batch_size, num_nodes, features]
        # adj_matrix: [num_nodes, num_nodes]
        
        # Normalize adjacency matrix (D^-1/2 * A * D^-1/2)
        D = tf.reduce_sum(adj_matrix, axis=1)
        D_sqrt_inv = tf.math.pow(D, -0.5)
        D_sqrt_inv = tf.where(tf.math.is_inf(D_sqrt_inv), tf.zeros_like(D_sqrt_inv), D_sqrt_inv)
        D_mat_inv_sqrt = tf.linalg.diag(D_sqrt_inv)
        normalized_adj = tf.matmul(tf.matmul(D_mat_inv_sqrt, adj_matrix), D_mat_inv_sqrt)
        
        # Graph convolution
        supports = tf.matmul(inputs, self.kernel)
        output = tf.matmul(normalized_adj, supports)
        output = output + self.bias
        
        if self.activation is not None:
            output = self.activation(output)
        
        return output

# DCRNN Model (Diffusion Convolutional Recurrent Neural Network)
def build_dcrnn_model(input_shape, adj_matrix, hidden_size=64, num_gnn_layers=2, num_rnn_layers=1, dropout=0.1):
    """Build a simplified DCRNN model"""
    inputs = Input(shape=input_shape)
    
    # Reshape inputs to [batch_size, num_nodes, features, time_steps]
    # For simplicity, we'll treat each feature as a separate node
    batch_size, time_steps, features = input_shape
    num_nodes = features
    x = Reshape((num_nodes, 1, time_steps))(inputs)
    
    # Transpose to [batch_size, time_steps, num_nodes, features]
    x = tf.transpose(x, perm=[0, 3, 1, 2])
    
    # GRU with graph convolution
    for i in range(num_rnn_layers):
        # Initialize hidden state
        h = tf.zeros_like(x[:, 0, :, :])
        outputs = []
        
        # Process each time step
        for t in range(time_steps):
            x_t = x[:, t, :, :]
            
            # Update gate
            z = Dense(hidden_size, activation='sigmoid')(x_t)
            
            # Reset gate
            r = Dense(hidden_size, activation='sigmoid')(x_t)
            
            # Candidate hidden state
            h_tilde = Dense(hidden_size, activation='tanh')(x_t * r)
            
            # New hidden state
            h = (1 - z) * h + z * h_tilde
            
            # Apply graph convolution to hidden state
            for _ in range(num_gnn_layers):
                h = Lambda(lambda x: GraphConvLayer(hidden_size, activation='relu')(x, adj_matrix))(h)
            
            outputs.append(h)
        
        # Stack outputs
        x = tf.stack(outputs, axis=1)
    
    # Final processing
    x = x[:, -1, :, :]  # Take the last time step
    x = tf.reduce_mean(x, axis=1)  # Average across nodes
    x = Dense(hidden_size//2, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# ASTGCN Model (Attention-based Spatial-Temporal Graph Convolutional Network)
def build_astgcn_model(input_shape, adj_matrix, hidden_size=64, num_heads=4, dropout=0.1):
    """Build a simplified ASTGCN model"""
    inputs = Input(shape=input_shape)
    
    # Reshape inputs to [batch_size, num_nodes, features, time_steps]
    batch_size, time_steps, features = input_shape
    num_nodes = features
    x = Reshape((num_nodes, 1, time_steps))(inputs)
    
    # Spatial attention
    spatial_attention = Dense(num_nodes, activation='softmax')(tf.reduce_mean(x, axis=-1))
    spatial_attention = tf.expand_dims(spatial_attention, axis=-1)
    
    # Apply spatial attention
    x_spatial = x * spatial_attention
    
    # Temporal attention
    temporal_attention = Dense(time_steps, activation='softmax')(tf.reduce_mean(x, axis=1))
    temporal_attention = tf.expand_dims(temporal_attention, axis=1)
    
    # Apply temporal attention
    x_temporal = x * temporal_attention
    
    # Combine spatial and temporal attention
    x = x_spatial + x_temporal
    
    # Graph convolution
    x = tf.transpose(x, perm=[0, 3, 1, 2])  # [batch_size, time_steps, num_nodes, features]
    
    for _ in range(2):  # 2 GCN layers
        x_gcn = []
        for t in range(time_steps):
            x_t = x[:, t, :, :]
            x_t = Lambda(lambda x: GraphConvLayer(hidden_size, activation='relu')(x, adj_matrix))(x_t)
            x_gcn.append(x_t)
        x = tf.stack(x_gcn, axis=1)
    
    # Final processing
    x = tf.reduce_mean(x, axis=2)  # Average across nodes
    x = GRU(hidden_size)(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# STGCN Model (Spatio-Temporal Graph Convolutional Network)
def build_stgcn_model(input_shape, adj_matrix, hidden_size=64, kernel_size=3, dropout=0.1):
    """Build a simplified STGCN model"""
    inputs = Input(shape=input_shape)
    
    # Reshape inputs to [batch_size, time_steps, num_nodes, features]
    batch_size, time_steps, features = input_shape
    num_nodes = features
    x = Reshape((time_steps, num_nodes, 1))(inputs)
    
    # Temporal convolution
    x_temporal = Conv1D(filters=hidden_size, kernel_size=kernel_size, padding='same')(inputs)
    x_temporal = LayerNormalization()(x_temporal)
    x_temporal = Activation('relu')(x_temporal)
    x_temporal = Reshape((time_steps, num_nodes, hidden_size))(x_temporal)
    
    # Spatial graph convolution
    x_spatial = []
    for t in range(time_steps):
        x_t = x_temporal[:, t, :, :]
        x_t = Lambda(lambda x: GraphConvLayer(hidden_size, activation='relu')(x, adj_matrix))(x_t)
        x_spatial.append(x_t)
    x = tf.stack(x_spatial, axis=1)
    
    # Another temporal convolution
    x = tf.reshape(x, [-1, time_steps, num_nodes * hidden_size])
    x = Conv1D(filters=hidden_size, kernel_size=kernel_size, padding='same')(x)
    x = LayerNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    
    # Final processing
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

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
    
    # Create a demo adjacency matrix (treating each feature as a node)
    num_features = len(exog_cols)
    adj_matrix = create_demo_adjacency_matrix(num_features)
    
    # 2) TRAIN MODELS
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # DCRNN Model
    print("\nTraining DCRNN Model...")
    dcrnn_model = build_dcrnn_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), adj_matrix=adj_matrix)
    dcrnn_model.fit(
        X_train_seq, y_train_seq,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # ASTGCN Model
    print("\nTraining ASTGCN Model...")
    astgcn_model = build_astgcn_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), adj_matrix=adj_matrix)
    astgcn_model.fit(
        X_train_seq, y_train_seq,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # STGCN Model
    print("\nTraining STGCN Model...")
    stgcn_model = build_stgcn_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), adj_matrix=adj_matrix)
    stgcn_model.fit(
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
    dcrnn_preds = rolling_forecast(dcrnn_model)
    astgcn_preds = rolling_forecast(astgcn_model)
    stgcn_preds = rolling_forecast(stgcn_model)
    
    # 4) EVALUATE AND PLOT
    evaluate_forecast("DCRNN", y_test, dcrnn_preds)
    evaluate_forecast("ASTGCN", y_test, astgcn_preds)
    evaluate_forecast("STGCN", y_test, stgcn_preds)
    
    # Plot results
    plt.figure(figsize=(14, 8))
    
    plt.plot(train_df['Date'], train_df[target_col], label='Train Actual', color='blue', marker='o')
    plt.plot(test_df['Date'], test_df[target_col], label='Test Actual', color='black', marker='o')
    plt.plot(test_df['Date'], dcrnn_preds, label='DCRNN Prediction', color='red', marker='x')
    plt.plot(test_df['Date'], astgcn_preds, label='ASTGCN Prediction', color='green', marker='+')
    plt.plot(test_df['Date'], stgcn_preds, label='STGCN Prediction', color='purple', marker='*')
    
    plt.axvline(x=test_df['Date'].iloc[0], color='gray', linestyle='--', label='Train-Test Split')
    plt.title("Spatio-Temporal GNN Models Comparison: Actual vs. Predicted")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()