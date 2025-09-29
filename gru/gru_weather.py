import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
from datetime import timedelta
from os.path import dirname, realpath
import json

import time
# Define the base model file name
base_model_file = dirname(__file__) + "\\gru_prediction_model_app_{}.h5"

# Load the data from the "output_weather.csv" file without a header (same as LSTM)
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

def create_sequences(data, seq_length, pred_horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_horizon + 1):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length : i + seq_length + pred_horizon])
    return np.array(X), np.array(y)

for target_application_id in target_application_ids:
    app_name = id_to_app_name.get(target_application_id)
    if app_name is None:
        print(f"Warning: Application ID {target_application_id} not found in mapping.")
        continue

    print(f"\n--- Predicting for Application: {app_name} (ID: {target_application_id}) ---")
    model_file = base_model_file.format(target_application_id)

    # Aggregate to daily totals for 7-day prediction (like LSTM)
    app_data = df[df['aplication_name'] == target_application_id].set_index('timestamp')['total'].resample('D').sum().fillna(0)

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(app_data.values.reshape(-1, 1))

    # Check if there's enough data to create sequences and for prediction
    if len(scaled_data) >= sequence_length + prediction_horizon:
        X, y = create_sequences(scaled_data, sequence_length, prediction_horizon)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        data_available = True
    else:
        print(f"Not enough daily data to create training sequences for Application {app_name} (ID: {target_application_id}). Need at least {sequence_length + prediction_horizon} days, got {len(scaled_data)}.")
        gru_model = None
        data_available = False

    # Build and train the GRU model
    if os.path.exists(model_file):
        print(f"Loading existing model from '{model_file}'...")
        gru_model = tf.keras.models.load_model(model_file, compile=False)
        # Recompile for further training or evaluation
        gru_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    else:
        if data_available:
            print(f"Creating and training a new model for Application {app_name}...")
            start_time = time.time()
            gru_model = tf.keras.models.Sequential([
                tf.keras.layers.GRU(50, activation='relu', input_shape=(sequence_length, 1)),
                tf.keras.layers.Dense(prediction_horizon)
            ])
            gru_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
            gru_model.fit(X_train, y_train, epochs=10000, batch_size=32, validation_split=0.1, verbose=0)
            # Keras warning: HDF5 is legacy, but we keep .h5 for compatibility. To use new format, change to .keras
            gru_model.save(model_file)
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"Trained GRU model saved to '{model_file}' for Application {app_name}")
            print(f"Model training completed in {elapsed:.2f} seconds.")
        else:
            gru_model = None
            print(f"Skipping model creation and training for Application {app_name} due to insufficient data.")

    # Predict the last 7 days of available data for MAE/MSE calculation (walk-forward validation)
    if gru_model is not None and len(scaled_data) >= sequence_length + 7:
        all_predictions = []
        prediction_dates = []
        for i in range(7):
            # Use the sequence ending right before the i-th last day
            seq_start = -(7 + sequence_length - i)
            seq_end = -(7 - i)
            if seq_start == 0:
                last_sequence = scaled_data[:seq_end].reshape(1, sequence_length, 1)
            else:
                last_sequence = scaled_data[seq_start:seq_end].reshape(1, sequence_length, 1)
            prediction_scaled = gru_model.predict(last_sequence)[0][0]
            all_predictions.append(prediction_scaled)
            prediction_dates.append(app_data.index[-7 + i])

        predictions = scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1)).flatten()
        predicted_series = pd.Series(predictions, index=pd.to_datetime(prediction_dates))

        print(f"\nPredictions for the last 7 days for Application {app_name} (ID: {target_application_id}):")
        for date, pred_val in predicted_series.items():
            print(f"{date.date()}: Predicted={pred_val:.2f}")

        # Calculate MAE and MSE for the last 7 days
        real_df = app_data.iloc[-7:]
        pred_df = predicted_series.loc[real_df.index]
        if not real_df.empty and len(real_df) == len(pred_df):
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            mae = mean_absolute_error(real_df.values, pred_df.values)
            mse = mean_squared_error(real_df.values, pred_df.values)
            print(f"\nMean Absolute Error (MAE): {mae:.2f}")
            print(f"Mean Squared Error (MSE): {mse:.2f}")
        else:
            print("No overlapping real values available to compute MAE/MSE for predictions.")
    else:
        print(f"Could not make predictions for Application {app_name} (ID: {target_application_id}) due to missing model or insufficient data.")