import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import timedelta
from os.path import dirname, realpath
import time

# Set up file paths
dirName = dirname(__file__)
csv_file = os.path.join(dirname(realpath(dirName)), "original data", "north2.csv")

# Read CSV and parse timestamp
df = pd.read_csv(csv_file, encoding='utf-8')
df['timestamp'] = pd.to_datetime(df['Data'] + ' ' + df['Hora'], errors='coerce')
df = df.sort_values('timestamp')

# Use only the precipitation column
value_col = "PRECIPITACAO TOTAL, HORARIO (mm)"
df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
df[value_col] = df[value_col].replace(-9999, np.nan)
df[value_col] = df[value_col].clip(lower=0)
df[value_col] = df[value_col].fillna(0)


# Use hourly data (no daily resampling)
app_data = df.set_index('timestamp')[value_col].fillna(0)

# Remove duplicate timestamps to avoid reindex error
app_data = app_data[~app_data.index.duplicated(keep='first')]

# Data quality checks before scaling
print(f"Data quality: min={app_data.min()}, max={app_data.max()}, any NaN={app_data.isna().any()}")
print(f"First 5 values before scaling: {app_data.values[:5]}")

# Always fit scaler on the full available data (including the last prediction window)
scaler = MinMaxScaler()
scaler.fit(app_data.values.reshape(-1, 1))
scaled_data = scaler.transform(app_data.values.reshape(-1, 1))
print(f"First 5 values after scaling: {scaled_data[:5].flatten()}")
print(f"Scaled data min={scaled_data.min()}, max={scaled_data.max()}, any NaN={np.isnan(scaled_data).any()}")




# Sequence and prediction horizon (hourly, matching LSTM)
sequence_length_hours = 24  # Look back 24 hours
prediction_horizon_hours = 168  # Predict the next 168 hours (7 days)
print(f"Using sequence_length_hours={sequence_length_hours}, prediction_horizon_hours={prediction_horizon_hours}")

def create_sequences_hours(data, seq_length, pred_horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_horizon + 1):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length : i + seq_length + pred_horizon])
    return np.array(X), np.array(y)

# Check if there's enough data to create sequences for hourly prediction
if len(scaled_data) > sequence_length_hours + prediction_horizon_hours - 1:
    X, y = create_sequences_hours(scaled_data, sequence_length_hours, prediction_horizon_hours)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    data_available = True
else:
    print(f"Not enough data to create training sequences.")
    autoencoder = None
    X_train, y_train = None, None
    data_available = False

# Build and train the Autoencoder model for daily prediction
model_file = os.path.join(dirName, "autoencoder_precipitacao_model_hourly.h5")
if os.path.exists(model_file):
    print(f"Loading existing model from '{model_file}'...")
    try:
        autoencoder = tf.keras.models.load_model(model_file, custom_objects={'mse': tf.keras.losses.MeanSquaredError})
    except Exception as e:
        print(f"Error loading model: {e}")
        autoencoder = None
else:
    if X_train is not None:
        print(f"Creating and training a new Autoencoder model for hourly precipitation prediction...")
        input_dim = X_train.shape[1]  # Sequence length (hours)
        output_dim = y_train.shape[1] # Prediction horizon (hours)
        encoding_dim = max(2, int(input_dim / 2))

        autoencoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(encoding_dim, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.RepeatVector(output_dim),
            tf.keras.layers.LSTM(encoding_dim, activation='relu', return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
        ])
        autoencoder.compile(optimizer='adam', loss='mse')
        start_time = time.time()
        history = autoencoder.fit(X_train, y_train, epochs=1, batch_size=8, validation_split=0.1, verbose=0)
        # Print loss history summary
        print(f"Training loss: min={np.min(history.history['loss'])}, max={np.max(history.history['loss'])}")
        if 'val_loss' in history.history:
            print(f"Validation loss: min={np.min(history.history['val_loss'])}, max={np.max(history.history['val_loss'])}")
        end_time = time.time()
        elapsed = end_time - start_time
        autoencoder.save(model_file)
        print(f"Trained Autoencoder model saved to '{model_file}'")
        print(f"Model training completed in {elapsed:.2f} seconds.")
    else:
        autoencoder = None
        print(f"Skipping model creation and training due to insufficient data.")

# Walk-forward validation: predict the next 7 days and compare to real values
if autoencoder is not None and len(scaled_data) > sequence_length_hours + prediction_horizon_hours:
    # Use the last available sequence to predict the next 24 hours
    last_sequence = scaled_data[-(sequence_length_hours + prediction_horizon_hours):-prediction_horizon_hours].reshape(1, sequence_length_hours)
    pred_scaled = autoencoder.predict(last_sequence)[0].flatten()
    print(f"First 5 predictions (scaled): {pred_scaled[:5]}")
    predictions = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    print(f"First 5 predictions (inversed): {predictions[:5]}")
    # Build datetimes for the prediction window
    pred_datetimes = app_data.index[-prediction_horizon_hours:]
    pred_series = pd.Series(predictions, index=pred_datetimes)
    # Get real values for those dates
    real_series = app_data.reindex(pred_series.index)
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mask = ~real_series.isna()
    if mask.sum() > 0:
        mae = mean_absolute_error(real_series[mask], pred_series[mask])
        mse = mean_squared_error(real_series[mask], pred_series[mask])
        print(f"\nPrediction for next {prediction_horizon_hours} hours:")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
    else:
        print("No overlapping real values available to compute MAE/MSE for prediction.")
else:
    print(f"Could not make predictions due to missing model or insufficient data.")