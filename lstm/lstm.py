import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adagrad
from sklearn.preprocessing import MinMaxScaler
import joblib
import re

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINED_MODELS_DIR = os.path.join(BASE_DIR, "trained_models")

os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)

FORECAST_HORIZON = 168  # Change this to 168 for 7 days, or any other value as needed
# --- Helper Functions ---
def create_lstm_model(input_shape):
    model = Sequential()
    # LSTM layers with BatchNorm and Dropout
    model.add(LSTM(192, input_shape=input_shape, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(96, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    # Dense layers with BatchNorm and reduced L1 regularization
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.00005)))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l1(0.00005)))
    model.add(BatchNormalization())
    # Output: forecast_horizon steps for the first feature
    model.add(Dense(FORECAST_HORIZON, activation='linear', kernel_regularizer=regularizers.l1(0.00001)))
    # Use Adagrad optimizer with learning_rate=0.001 and MSE loss
    optimizer = Adagrad(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def create_sequences(data, seq_length):
    # For direct multi-step forecasting: y is the next FORECAST_HORIZON steps
    xs, ys = [], []
    for i in range(len(data) - seq_length - FORECAST_HORIZON + 1):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length:i+seq_length+FORECAST_HORIZON, 0])  # predict first feature (target) only
    return np.array(xs), np.array(ys)

# --- Training Function ---
def train_lstm(csv_path, mappings_path, model_id, model_path):
    df = pd.read_csv(csv_path)
    with open(mappings_path, 'r') as f:
        mappings = json.load(f)
    # Assume first column is timestamp, rest are features
    df = df.sort_values(df.columns[0])
    # Ensure value_column is the first feature after timestamp
    value_column = None
    if isinstance(mappings, dict) and "value_column" in mappings:
        value_column = mappings["value_column"]
    else:
        value_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    if value_column not in df.columns:
        raise ValueError(f"Selected value column '{value_column}' not found in CSV columns: {list(df.columns)}")
    print(f"[DEBUG] Training: value_column set to '{value_column}'", file=sys.stderr)
    # Reorder columns: timestamp, value_column, then the rest
    timestamp_col = df.columns[0]
    other_cols = [col for col in df.columns if col not in [timestamp_col, value_column]]
    df = df[[timestamp_col, value_column] + other_cols]
    X = df.iloc[:, 1:].values.astype(np.float32)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    seq_length = 24  # 24 hours (change as needed)
    print(f"[DEBUG] Training: DataFrame columns after reordering: {list(df.columns)}", file=sys.stderr)
    X_seq, y_seq = create_sequences(X_scaled, seq_length)
    # Check if there is enough data to create at least one training sample
    if X_seq.size == 0 or y_seq.size == 0:
        raise ValueError(f"Not enough data to create training samples: need at least {seq_length + FORECAST_HORIZON} rows, got {len(X_scaled)}.")
    model = create_lstm_model((seq_length, X_seq.shape[2] if X_seq.ndim == 3 else X_seq.shape[1]))
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X_seq, y_seq,
        epochs=100,
        batch_size=16,
        verbose=2,
        validation_split=0.2,
        shuffle=True,
        callbacks=[early_stop]
    )
    # Save model and scaler
    os.makedirs(model_path, exist_ok=True)
    model.save(os.path.join(model_path, f"model_{model_id}.keras"))
    # Save the full scaler object for inverse_transform
    joblib.dump(scaler, os.path.join(model_path, "scaler.save"))
    # Save the selected value column name as passed from the frontend/backend (should be in mappings or a config)
    # If a value_column is specified in mappings, use it; otherwise, default to first after timestamp
    value_column = None
    if isinstance(mappings, dict) and "value_column" in mappings:
        value_column = mappings["value_column"]
    else:
        value_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    if value_column not in df.columns:
        raise ValueError(f"Selected value column '{value_column}' not found in CSV columns: {list(df.columns)}")
    with open(os.path.join(model_path, "source_columns.json"), 'w') as f:
        json.dump({"columns": list(df.columns), "value_column": value_column}, f)
    # Save last sequence for prediction
    np.save(os.path.join(model_path, "last_sequence.npy"), X_scaled[-seq_length:])

# --- Prediction Function ---
def predict_lstm(model_id, model_path, t1=None, t2=None):
    import sys
    model_file = os.path.join(model_path, f"model_{model_id}.keras")
    scaler_file = os.path.join(model_path, "scaler.save")
    last_seq_file = os.path.join(model_path, "last_sequence.npy")
    columns_file = os.path.join(model_path, "source_columns.json")
    model = keras.models.load_model(model_file)
    scaler = joblib.load(scaler_file)
    last_seq = np.load(last_seq_file)
    # Load feature columns from preprocessed CSV (excluding timestamp)
    import csv
    preproc_csv = None
    # Extract timestamp from model_path (assumes model_path ends with timestamp)
    match = re.search(r'(\d{8}_\d{6})', model_path)
    # Try to find the preprocessed CSV file by timestamp
    if match:
        timestamp = match.group(1)
        preproc_dir = os.path.join(os.path.dirname(os.path.dirname(model_path)), 'preprocessed_data')
        for fname in os.listdir(preproc_dir):
            if fname.startswith('output_' + timestamp) and fname.endswith('.csv'):
                preproc_csv = os.path.join(preproc_dir, fname)
                break
    if preproc_csv is None:
        raise RuntimeError("Could not find preprocessed CSV for prediction column order.")
    with open(preproc_csv, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
    feature_columns = [col for col in header if col != 'timestamp']
    # Load the selected value column name from source_columns.json
    with open(columns_file, 'r') as f:
        source_info = json.load(f)
        value_column = source_info.get("value_column", feature_columns[0])
    # Reorder feature_columns so value_column is first, matching training order
    if value_column in feature_columns:
        other_cols = [col for col in feature_columns if col != value_column]
        feature_columns = [value_column] + other_cols
    if value_column not in feature_columns:
        raise RuntimeError(f"Target value column '{value_column}' not found in features: {feature_columns}. Please ensure the selected value column matches the preprocessed CSV.")
    columns = feature_columns
    # Use the timestamp from the model_path to select the correct mappings file
    mappings_dir = os.path.join(os.path.dirname(os.path.dirname(model_path)), 'preprocessed_data')
    # Try to find the mappings file by timestamp
    mappings = None
    if match:
        timestamp = match.group(1)
        for fname in os.listdir(mappings_dir):
            if fname.startswith('mappings_' + timestamp) and fname.endswith('.json'):
                mappings_file = os.path.join(mappings_dir, fname)
                with open(mappings_file, 'r') as mf:
                    mappings = json.load(mf)
                break
    if mappings is None:
        # fallback to mappings.json if timestamped not found
        default_mappings_file = os.path.join(mappings_dir, 'mappings.json')
        if os.path.exists(default_mappings_file):
            with open(default_mappings_file, 'r') as mf:
                mappings = json.load(mf)
        else:
            mappings = {}

    # Compute low_activity_threshold, T1, T2 if possible (use percentiles of last N time steps)
    low_activity_threshold = None
    try:
        df = pd.read_csv(preproc_csv)
        if value_column in df.columns:
            last_n = min(168, len(df))
            values = df[value_column].tail(last_n)
            low_activity_threshold = float(np.percentile(values, 10))
            # Calculate T1 and T2 if not provided
            if t1 is None:
                t1 = float(np.percentile(values, 95))
            if t2 is None:
                t2 = float(np.percentile(values, 99.5))
    except Exception as e:
        import sys
        print(f"[DEBUG] Could not compute thresholds: {e}", file=sys.stderr)

    seq = last_seq.copy()
    preds = []
    now = datetime.now().replace(minute=0, second=0, microsecond=0)

    # Direct multi-step prediction: predict next FORECAST_HORIZON steps at once
    pred = model.predict(seq[np.newaxis, :, :], verbose=0)[0]  # shape: (FORECAST_HORIZON,)
    # Inverse transform: need to reconstruct full feature vector for scaler
    # Find the target column (the value column used for prediction)
    # Use the value_column saved during training
    if value_column not in feature_columns:
        raise RuntimeError(f"Target value column '{value_column}' not found in features: {feature_columns}")
    target_col = value_column
    target_col_idx = feature_columns.index(target_col)
    # Use last-seen values for non-target columns in inverse_transform
    last_seen = seq[-1]  # shape: (num_features,)
    for i in range(FORECAST_HORIZON):
        dummy = last_seen.copy()
        dummy[target_col_idx] = pred[i]  # set only the target feature
        dummy = dummy.reshape(1, -1)
        pred_orig = scaler.inverse_transform(dummy)[0]
        # Debug: print predicted value and inverse transformed vector
        print(f"[DEBUG] Step {i}: pred (scaled)={pred[i]}, pred_orig={pred_orig}", file=sys.stderr)
        mapped_columns = feature_columns.copy()
        if mappings:
            mapped_columns = [mappings.get(col, col) for col in mapped_columns]
        col_str = str(mapped_columns)
        predicted_value = float(pred_orig[target_col_idx])
        entry = {
            "timestamp": (now + timedelta(hours=i)).isoformat(),
            "predicted_value": predicted_value,
            "source_columns": col_str,
            "anomaly_status": None
        }
        # New anomaly status logic
        print(f"[DEBUG] T1: {t1}", file=sys.stderr)
        print(f"[DEBUG] T2: {t2}", file=sys.stderr)
        print(f"[DEBUG] low_activity_threshold: {low_activity_threshold}", file=sys.stderr)    
        if t2 is not None and predicted_value > t2:
            entry["anomaly_status"] = "critical"
        elif t1 is not None and predicted_value > t1:
            entry["anomaly_status"] = "warning"
        elif low_activity_threshold is not None and predicted_value >= low_activity_threshold:
            entry["anomaly_status"] = "normal"
        elif low_activity_threshold is not None and predicted_value < low_activity_threshold:
            entry["anomaly_status"] = "maintenance"
        else:
            entry["anomaly_status"] = "maintenance"
        print(f"[DEBUG] entry: {entry}", file=sys.stderr)
        preds.append(entry)
    print(json.dumps(preds))
    return preds

# --- CLI Entrypoint ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LSTM Model CLI")
    parser.add_argument("mode", choices=["train", "predict"], help="Mode: train or predict")
    parser.add_argument("csv_path", nargs="?", help="CSV path for training")
    parser.add_argument("mappings_path", nargs="?", help="Mappings path for training")
    parser.add_argument("model_id", help="Model ID")
    parser.add_argument("model_path", help="Model path")
    parser.add_argument("--t1", type=float, default=None, help="Anomaly threshold 1")
    parser.add_argument("--t2", type=float, default=None, help="Anomaly threshold 2")

    args = parser.parse_args()
    if args.mode == "train":
        if not (args.csv_path and args.mappings_path and args.model_id and args.model_path):
            print("Usage: python lstm.py train <csv_path> <mappings_path> <model_id> <model_path>", file=sys.stderr)
            sys.exit(1)
        train_lstm(args.csv_path, args.mappings_path, args.model_id, args.model_path)
    elif args.mode == "predict":
        if not (args.model_id and args.model_path):
            print("Usage: python lstm.py predict <model_id> <model_path> [--t1 <float>] [--t2 <float>]", file=sys.stderr)
            sys.exit(1)
        predict_lstm(args.model_id, args.model_path, t1=args.t1, t2=args.t2)
