import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
import os
from datetime import timedelta
from os.path import dirname, realpath
import json
import time

# Define a threshold for anomaly detection (you'll need to tune this)
anomaly_threshold = 0.03  # Example threshold

# Define the base model file name
base_model_file = dirname(__file__) + "\\ocsvm_anomaly_model_app_{}.h5"

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

def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i : i + seq_length])
    return np.array(X)

for target_application_id in target_application_ids:
    app_name = id_to_app_name.get(target_application_id)
    if app_name is None:
        print(f"Warning: Application ID {target_application_id} not found in mapping.")
        continue

    print(f"\n--- Anomaly Detection for Application: {app_name} (ID: {target_application_id}) using OCSVM ---")

    # Consider only the current application
    # Aggregate by day (ignore hours)
    app_data = df[df['aplication_name'] == target_application_id].set_index('timestamp')['total'].resample('D').sum().fillna(0)

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(app_data.values.reshape(-1, 1))

    # Update sequence length for daily prediction
    sequence_length_days = 7  # Look back 7 days
    prediction_days = 7       # Predict the next 7 days

    # Redefine create_sequences for daily prediction
    def create_sequences_days(data, seq_length):
        X = []
        for i in range(len(data) - seq_length + 1):
            X.append(data[i : i + seq_length])
        return np.array(X)

    # Create sequences for the OCSVM (daily)
    sequences = create_sequences_days(scaled_data, sequence_length_days)

    # Hold out the last 7 days as test set (label as normal)
    # Synthetic test sample generation block
    if len(sequences) > 0:
        train_sequences = sequences
        # Calculate mean and std for each position in the sequence (across all training samples)
        mean_per_pos = np.mean(train_sequences, axis=0).flatten()  # shape: (sequence_length_days,)
        std_per_pos = np.std(train_sequences, axis=0).flatten()    # shape: (sequence_length_days,)
        # Generate 7 synthetic sequences (each is a sequence of 7 days)
        synthetic_sequences = []
        y_true = []
        rng = np.random.default_rng()
        for _ in range(7):
            # Randomly decide if this sample is normal or anomaly
            is_anomaly = rng.random() < 0.5  # 50% chance anomaly
            if is_anomaly:
                # Generate anomaly with less extreme noise (harder to detect)
                anomaly = rng.normal(loc=mean_per_pos, scale=std_per_pos * 2)
                synthetic_sequences.append(anomaly)
                y_true.append(-1)
            else:
                normal = rng.normal(loc=mean_per_pos, scale=std_per_pos)
                synthetic_sequences.append(normal)
                y_true.append(1)
        test_sequences = np.array(synthetic_sequences)
        y_true = np.array(y_true, dtype=int)
        print("Synthetic test samples:")
        for i, (sample, label) in enumerate(zip(test_sequences, y_true)):
            print(f"Sample {i+1}: label={'anomaly' if label==-1 else 'normal'}, values={sample}")
        data_available = True
    else:
        print(f"Not enough data to create sequences for Application {app_name} (ID: {target_application_id}).")
        train_sequences, test_sequences, y_true = None, None, None
        data_available = False

    # Build and train the OCSVM model
    if data_available and train_sequences is not None and len(train_sequences) > 0:
        print(f"Training OCSVM model for Application {app_name}...")
        start_time = time.time()
        X_train = train_sequences.reshape(train_sequences.shape[0], -1)
        # Use synthetic test set if available, else fallback to last 7 days
        # Always use synthetic test set for evaluation
        X_test = test_sequences.reshape(test_sequences.shape[0], -1)
        y_test_labels = y_true
        ocsvm = OneClassSVM(gamma=0.01, nu=0.1)
        ocsvm.fit(X_train)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"OCSVM model training completed in {elapsed:.2f} seconds.")
        # Evaluate on test set if available
        if X_test is not None and y_test_labels is not None:
            y_pred = ocsvm.predict(X_test)
            n_anomalies = np.sum(y_pred == -1)
            print(f"Detected {n_anomalies} anomalies out of {len(y_pred)} test samples.")
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            acc = accuracy_score(y_test_labels, y_pred)
            prec = precision_score(y_test_labels, y_pred, pos_label=-1, zero_division=0)
            rec = recall_score(y_test_labels, y_pred, pos_label=-1, zero_division=0)
            f1 = f1_score(y_test_labels, y_pred, pos_label=-1, zero_division=0)
            print(f"Accuracy: {acc:.4f}")
            print(f"Precision (anomaly): {prec:.4f}")
            print(f"Recall (anomaly): {rec:.4f}")
            print(f"F1-score (anomaly): {f1:.4f}")
            print(f"True labels (y_true): {y_test_labels}")
            print(f"Predicted labels (y_pred): {y_pred}")
            # Print decision function scores for each test sample
            scores = ocsvm.decision_function(X_test)
        else:
            print("No test set available for evaluation.")
        print(f"Training time: {elapsed:.2f} seconds")
    else:
        ocsvm = None
        print(f"Skipping model creation and training for Application {app_name} due to insufficient data.")