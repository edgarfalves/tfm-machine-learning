import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import timedelta
from os.path import dirname, realpath
import json
import time

# Define the base model file name
base_model_file = dirname(__file__) + "\\autoencoder_prediction_model_app_{}.h5"

# Load the data from the "output.csv" file without a header
dirName = dirname(__file__)
csv_file = dirname(realpath(dirName)) + "\\preprocessed_data\\output_weather.csv"

column_names = ["date", "hour", "aplication_name", "total"]
df = pd.read_csv(csv_file, header=None, names=column_names)

# Load application name mapping from JSON file
json_file_path = dirname(realpath(dirName))+"\\preprocessed_data\\mappings.json"
try:
    with open(json_file_path, 'r') as f:
        all_mappings = json.load(f)
        application_mapping = all_mappings.get('aplication_name', {})
    # The keys in application_mapping are the names, the values are the IDs
    app_name_to_id = application_mapping
    id_to_app_name = {v: k for k, v in app_name_to_id.items()}
    target_application_ids = list(app_name_to_id.values())
except FileNotFoundError:
    print(f"Error: '{json_file_path}' not found. Using default target applications [0, 1, 2, 3].")
    target_application_ids = [0, 1, 2, 3]
    id_to_app_name = {i: str(i) for i in target_application_ids} # Default mapping
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{json_file_path}'. Using default target applications [0, 1, 2, 3].")
    target_application_ids = [0, 1, 2, 3]
    id_to_app_name = {i: str(i) for i in target_application_ids} # Default mapping

# Convert 'date' and 'hour' to a datetime format suitable for time series
df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['hour'].astype(str) + ':00:00')
df = df.sort_values('timestamp')

# Define sequence length and prediction horizon
sequence_length = 24 * 7  # Look back 7 days (24 hours/day)
prediction_horizon = 24  # Predict the next 24 hours
prediction_days = 7

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - prediction_horizon + 1):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length : i + seq_length + prediction_horizon])
    return np.array(X), np.array(y)

for target_application_id in target_application_ids:
    app_name = id_to_app_name.get(target_application_id)
    if app_name is None:
        print(f"Warning: Application ID {target_application_id} not found in mapping.")
        continue

    print(f"\n--- Predicting for Application: {app_name} (ID: {target_application_id}) using Autoencoder ---")
    model_file = base_model_file.format(target_application_id)

    # Debug: Show unique application names/IDs and sample of DataFrame
    print(f"Unique aplication_name values in CSV: {df['aplication_name'].unique()}")
    print(f"First 5 rows of CSV:\n{df.head()}")

    # Consider only the current application
    filtered = df[df['aplication_name'] == target_application_id]
    print(f"First 5 rows for application {target_application_id} (should match app_name {app_name}):\n{filtered.head()}")
    if not filtered.empty:
        print(f"First 5 values of 'total' column for this app: {filtered['total'].values[:5]}")
    else:
        print(f"No data found for application ID {target_application_id}!")
    # Aggregate by day (ignore hours)
    app_data = filtered.set_index('timestamp')['total'].resample('D').sum().fillna(0)

    # Data quality checks before scaling
    print(f"Data quality for {app_name}: min={app_data.min()}, max={app_data.max()}, any NaN={app_data.isna().any()}")
    print(f"First 5 values before scaling: {app_data.values[:5]}")

    # Always fit scaler on the full available data (including the last prediction window)
    scaler = MinMaxScaler()
    scaler.fit(app_data.values.reshape(-1, 1))
    scaled_data = scaler.transform(app_data.values.reshape(-1, 1))
    print(f"First 5 values after scaling: {scaled_data[:5].flatten()}")
    print(f"Scaled data min={scaled_data.min()}, max={scaled_data.max()}, any NaN={np.isnan(scaled_data).any()}")

    # Update sequence and prediction horizon for daily prediction
    sequence_length_days = 7  # Look back 7 days
    prediction_horizon_days = 7  # Predict the next 7 days

    # Redefine create_sequences for daily prediction
    def create_sequences_days(data, seq_length, pred_horizon):
        X, y = [], []
        for i in range(len(data) - seq_length - pred_horizon + 1):
            X.append(data[i : i + seq_length])
            y.append(data[i + seq_length : i + seq_length + pred_horizon])
        return np.array(X), np.array(y)

    # Check if there's enough data to create sequences for daily prediction
    if len(scaled_data) > sequence_length_days + prediction_horizon_days - 1:
        X, y = create_sequences_days(scaled_data, sequence_length_days, prediction_horizon_days)
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        data_available = True
    else:
        print(f"Not enough data to create training sequences for Application {app_name} (ID: {target_application_id}).")
        autoencoder = None
        X_train, y_train = None, None
        data_available = False

    # Build and train the Autoencoder model for daily prediction
    if os.path.exists(model_file):
        print(f"Loading existing model from '{model_file}'...")
        try:
            autoencoder = tf.keras.models.load_model(model_file, custom_objects={'mse': tf.keras.losses.MeanSquaredError})
        except Exception as e:
            print(f"Error loading model for Application {app_name}: {e}")
            autoencoder = None
    else:
        if X_train is not None:
            print(f"Creating and training a new Autoencoder model for Application {app_name} (daily prediction)...")
            input_dim = X_train.shape[1]  # Sequence length (days)
            output_dim = y_train.shape[1] # Prediction horizon (days)
            encoding_dim = max(2, int(input_dim / 2))

            autoencoder = tf.keras.models.Sequential([
                tf.keras.layers.Dense(encoding_dim, activation='relu', input_shape=(input_dim,)),
                tf.keras.layers.RepeatVector(output_dim),
                tf.keras.layers.LSTM(encoding_dim, activation='relu', return_sequences=True),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
            ])
            autoencoder.compile(optimizer='adam', loss='mse')
            start_time = time.time()
            history = autoencoder.fit(X_train, y_train, epochs=10000, batch_size=32, validation_split=0.1, verbose=0)
            # Print loss history summary
            print(f"Training loss: min={np.min(history.history['loss'])}, max={np.max(history.history['loss'])}")
            if 'val_loss' in history.history:
                print(f"Validation loss: min={np.min(history.history['val_loss'])}, max={np.max(history.history['val_loss'])}")
            end_time = time.time()
            elapsed = end_time - start_time
            autoencoder.save(model_file)
            print(f"Trained Autoencoder model saved to '{model_file}' for Application {app_name}")
            print(f"Model training completed in {elapsed:.2f} seconds.")
        else:
            autoencoder = None
            print(f"Skipping model creation and training for Application {app_name} due to insufficient data.")

    # Walk-forward validation: predict the next 7 days and compare to real values
    if autoencoder is not None and len(scaled_data) > sequence_length_days + prediction_horizon_days:
        # Use the last available sequence to predict the next 7 days
        last_sequence = scaled_data[-(sequence_length_days + prediction_horizon_days):-prediction_horizon_days].reshape(1, sequence_length_days)
        pred_scaled = autoencoder.predict(last_sequence)[0].flatten()
        print(f"First 5 predictions (scaled): {pred_scaled[:5]}")
        predictions = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        print(f"First 5 predictions (inversed): {predictions[:5]}")
        # Build datetimes for the prediction window
        pred_datetimes = app_data.index[-prediction_horizon_days:]
        pred_series = pd.Series(predictions, index=pred_datetimes)
        # Get real values for those dates
        real_series = app_data.reindex(pred_series.index)
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mask = ~real_series.isna()
        if mask.sum() > 0:
            mae = mean_absolute_error(real_series[mask], pred_series[mask])
            mse = mean_squared_error(real_series[mask], pred_series[mask])
            print(f"\nPrediction for next {prediction_horizon_days} days for Application {app_name}:")
            print(f"Mean Absolute Error (MAE): {mae:.2f}")
            print(f"Mean Squared Error (MSE): {mse:.2f}")
        else:
            print("No overlapping real values available to compute MAE/MSE for prediction.")
    else:
        print(f"Could not make predictions for Application {app_name} (ID: {target_application_id}) due to missing model or insufficient data.")