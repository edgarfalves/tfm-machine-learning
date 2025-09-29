import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from os.path import dirname, realpath
import json

# Real values for August 25–31, 2018
real_values = {
    "2018-08-25": 31.8000011,
    "2018-08-26": 29.3999996,
    "2018-08-27": 31.5,
    "2018-08-28": 28.8999996,
    "2018-08-29": 24.2000008,
    "2018-08-30": 26.0,
    "2018-08-31": 30.1000004
}

# Define the base model file name
base_model_file = dirname(__file__) + "\\lstm_prediction_model_app_{}.h5"

# Load the data
dirName = dirname(__file__)
csv_file = dirname(realpath(dirName)) + "\\preprocessed_data\\output_weather.csv"
column_names = ["date", "hour", "aplication_name", "total"]
df = pd.read_csv(csv_file, header=None, names=column_names)

# Load app name mappings
json_file_path = dirname(realpath(dirName)) + "\\preprocessed_data\\mappings.json"
try:
    with open(json_file_path, 'r') as f:
        all_mappings = json.load(f)
        application_mapping = all_mappings.get('aplication_name', {})
    app_name_to_id = application_mapping
    id_to_app_name = {v: k for k, v in app_name_to_id.items()}
    target_application_ids = list(app_name_to_id.values())
except (FileNotFoundError, json.JSONDecodeError):
    target_application_ids = [0]
    id_to_app_name = {0: "default"}

# Prepare timestamps and aggregate to daily
df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['hour'].astype(str) + ':00:00')
df = df.sort_values('timestamp')

sequence_length = 7  # 7 days
prediction_horizon = 1  # Predict next day
prediction_days = 7     # Predict 7 days ahead

def create_sequences(data, seq_length, pred_horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_horizon + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+pred_horizon])
    return np.array(X), np.array(y)

for target_application_id in target_application_ids:
    app_name = id_to_app_name.get(target_application_id, "unknown")
    model_file = base_model_file.format(target_application_id)

    print(f"\n--- Predicting for Application: {app_name} (ID: {target_application_id}) ---")
    
    # Resample to daily totals
    app_data = df[df['aplication_name'] == target_application_id].set_index('timestamp')['total'].resample('D').sum().fillna(0)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(app_data.values.reshape(-1, 1))

    if len(scaled_data) > sequence_length + prediction_horizon - 1:
        X, y = create_sequences(scaled_data, sequence_length, prediction_horizon)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        data_available = True
    else:
        print("Not enough data.")
        continue

    if os.path.exists(model_file):
        print(f"Loading model from {model_file}...")
        model = tf.keras.models.load_model(model_file, custom_objects={'mse': tf.keras.losses.MeanSquaredError})
    else:
        import time
        print("Training new model...")
        start_time = time.time()
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
            tf.keras.layers.Dense(prediction_horizon)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=10, verbose=1)
        model.save(model_file)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Model training completed in {elapsed:.2f} seconds.")

    last_sequence = scaled_data[-sequence_length:].reshape((1, sequence_length, 1))
    all_predictions = []

    for _ in range(prediction_days):
        prediction_scaled = model.predict(last_sequence)[0]
        all_predictions.append(prediction_scaled[0])
        new_sequence = np.concatenate([last_sequence[0][1:], prediction_scaled.reshape(-1, 1)], axis=0)
        last_sequence = new_sequence.reshape((1, sequence_length, 1))

    future_predictions = scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1)).flatten()

    start_date = app_data.index[-1] + pd.Timedelta(days=1)
    prediction_dates = [start_date + pd.Timedelta(days=i) for i in range(prediction_days)]

    predicted_series = pd.Series(future_predictions, index=pd.to_datetime(prediction_dates))
    real_series = pd.Series(real_values)
    real_series.index = pd.to_datetime(real_series.index)

    predicted_series.index = pd.to_datetime(predicted_series.index)
    common_dates = real_series.index.intersection(predicted_series.index)

    print("\nPredictions vs Real:")
    for date in common_dates:
        pred_val = predicted_series[date]
        real_val = real_series[date]
        print(f"{date.date()}: Predicted={pred_val:.2f}°C, Real={real_val:.2f}°C")

    if not common_dates.empty:
        mae = mean_absolute_error(real_series.loc[common_dates], predicted_series.loc[common_dates])
        mse = mean_squared_error(real_series.loc[common_dates], predicted_series.loc[common_dates])
        print(f"\nMean Absolute Error (MAE): {mae:.2f}°C")
        print(f"Mean Squared Error (MSE): {mse:.2f}°C²")
    else:
        print("No overlapping dates to compare predictions.")
