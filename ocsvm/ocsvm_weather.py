import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
import os
from datetime import timedelta
from os.path import dirname, realpath
import time
import joblib

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

# Use hourly data and resample to daily totals (OCSVM expects daily sequences here)
app_data = df.set_index('timestamp')[value_col].resample('D').sum().fillna(0)

# Remove duplicate timestamps if any
app_data = app_data[~app_data.index.duplicated(keep='first')]

print(f"Data quality: min={app_data.min()}, max={app_data.max()}, any NaN={app_data.isna().any()}")

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(app_data.values.reshape(-1, 1))

# Sequence settings (daily)
sequence_length_days = 7  # Look back 7 days

def create_sequences_days(data, seq_length):
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i : i + seq_length])
    return np.array(X)

# Create sequences
sequences = create_sequences_days(scaled_data, sequence_length_days)

if len(sequences) == 0:
    print("Not enough data to create sequences for OCSVM.")
    ocsvm = None
else:
    train_sequences = sequences
    # Calculate mean and std for each position in the sequence (across all training samples)
    mean_per_pos = np.mean(train_sequences, axis=0).flatten()
    std_per_pos = np.std(train_sequences, axis=0).flatten()

    # Generate synthetic test sequences (7 samples)
    synthetic_sequences = []
    y_true = []
    rng = np.random.default_rng()
    for _ in range(7):
        is_anomaly = rng.random() < 0.5
        if is_anomaly:
            anomaly = rng.normal(loc=mean_per_pos, scale=std_per_pos * 2)
            synthetic_sequences.append(anomaly)
            y_true.append(-1)
        else:
            normal = rng.normal(loc=mean_per_pos, scale=std_per_pos)
            synthetic_sequences.append(normal)
            y_true.append(1)
    test_sequences = np.array(synthetic_sequences)
    y_test = np.array(y_true, dtype=int)
    print("Synthetic test samples generated.")

    # Train OCSVM
    X_train = train_sequences.reshape(train_sequences.shape[0], -1)
    X_test = test_sequences.reshape(test_sequences.shape[0], -1)
    ocsvm = OneClassSVM(gamma=0.01, nu=0.1)
    start_time = time.time()
    ocsvm.fit(X_train)
    elapsed = time.time() - start_time
    print(f"OCSVM trained in {elapsed:.2f}s")

    # Evaluate on synthetic test set
    y_pred = ocsvm.predict(X_test)
    n_anoms = np.sum(y_pred == -1)
    print(f"Detected {n_anoms} anomalies out of {len(y_pred)} synthetic test samples")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=-1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=-1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=-1, zero_division=0)
    print(f"Accuracy: {acc:.4f}, Precision(anom): {prec:.4f}, Recall(anom): {rec:.4f}, F1(anom): {f1:.4f}")

    # Save model and scaler
    model_path = os.path.join(dirName, "ocsvm_model")
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(ocsvm, os.path.join(model_path, "ocsvm_model.joblib"))
    joblib.dump(scaler, os.path.join(model_path, "scaler.save"))
    print(f"Model and scaler saved to {model_path}")