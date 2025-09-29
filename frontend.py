import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"
PAGE_CONFIG = {"page_title": "Prediction Frontend", "layout": "wide"}

# --- Helper Functions ---
def get_models(model_type="lstm"):
    """Fetches available models from the API."""
    try:
        response = requests.get(f"{API_URL}/models/{model_type}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching models: {e}")
        return []

def delete_model_from_backend(model_name):
    """Deletes a model from the backend."""
    try:
        response = requests.delete(f"{API_URL}/models/{model_name}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error deleting model: {e}")
        return None

def fetch_and_display_predictions(model_name):
    """Fetches and displays pre-existing predictions for a model."""
    with st.spinner(f"Fetching predictions for model '{model_name}'..."):
        try:
            response = requests.get(f"{API_URL}/predictions/{model_name}")
            response.raise_for_status()
            predictions = response.json()
            st.session_state['predictions'] = {
                "model_version": model_name,
                "predictions": predictions
            }
            st.success(f"Found {len(predictions)} saved predictions for model '{model_name}'.")
        except requests.exceptions.RequestException as e:
            error_message = f"Error fetching predictions: {e}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    detail = e.response.json().get('detail', e.response.text)
                    error_message = f"Error fetching predictions: {detail}"
                except json.JSONDecodeError:
                    error_message = f"Error fetching predictions: Status {e.response.status_code} - {e.response.reason}"
            st.error(error_message)

# --- Main Application ---
def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(**PAGE_CONFIG)
    st.title("Time-Series Prediction Frontend")

    # --- Sidebar ---
    st.sidebar.header("Model Actions")

    # --- Model Selection and Viewing ---
    available_models = get_models()
    model_names = [model['model_name'] for model in available_models]
    
    selected_model = None
    if not model_names:
        st.sidebar.info("No models available. Upload a CSV to train a new model.")
    else:
        selected_model = st.sidebar.selectbox("Select an Existing Model", model_names)
        if st.sidebar.button("Show Saved Predictions"):
            fetch_and_display_predictions(selected_model)
        if selected_model:
            st.sidebar.subheader("Anomaly Detection Thresholds")
            t1 = st.sidebar.number_input("Anomaly Threshold (T1)", min_value=0.0, value=None, step=0.01, format="%.4f")
            t2 = st.sidebar.number_input("Anomaly Threshold (T2)", min_value=0.0, value=None, step=0.01, format="%.4f")
            if t1 is not None and t2 is not None and t1 >= t2:
                st.sidebar.error("T1 must be less than T2")
            st.session_state['t1'] = t1
            st.session_state['t2'] = t2
            if st.sidebar.button("Generate New Predictions with Selected Model"):
                run_prediction(selected_model, st.session_state['t1'], st.session_state['t2'])

    # --- File Upload Section ---
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV to Train", type=["csv"])
    
    df = None
    value_col, date_col, hour_col, other_cols = None, None, None, []

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.subheader("Column Selection")

        value_col = st.sidebar.selectbox("Select Value Column", df.columns)
        date_col = st.sidebar.selectbox("Select Date Column", df.columns)

        hour_col = None
        if pd.to_datetime(df[date_col], errors='coerce').dt.hour.sum() == 0:
            hour_col = st.sidebar.selectbox("Select Hour Column (Optional)", [None] + list(df.columns))

        # Only allow grouping by non-numeric columns
        candidate_group_cols = [c for c in df.columns if c not in [value_col, date_col, hour_col]]
        group_cols = st.sidebar.multiselect("Select Other Columns for Grouping", candidate_group_cols)

        # For each selected grouping column, if categorical, let user pick value(s) to keep
        group_col_values = {}
        selected_numeric_cols = []
        for col in group_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                selected_numeric_cols.append(col)
                continue
            unique_vals = sorted(df[col].dropna().unique())
            selected_val = st.sidebar.selectbox(f"Value for '{col}' to keep", unique_vals, key=f"val_{col}")
            group_col_values[col] = selected_val

        aggregation = st.sidebar.radio("Aggregation Method for Value Column", ["sum", "mean"], index=0, format_func=lambda x: "SUM" if x=="sum" else "AVERAGE")

        st.sidebar.subheader("Actions")
        if st.sidebar.button("Train New Model"):
            train_model(df, value_col, date_col, hour_col, group_col_values, aggregation, selected_numeric_cols)

    # --- Model Management ---
    st.sidebar.header("Manage Models")
    if model_names:
        model_to_delete = st.sidebar.selectbox("Select Model to Delete", model_names, key="delete_select")
        if st.sidebar.button("Delete Selected Model"):
            if delete_model_from_backend(model_to_delete):
                st.sidebar.success(f"Model '{model_to_delete}' deleted successfully.")
                st.session_state.pop('predictions', None)
                st.rerun()
    else:
        st.sidebar.info("No models to manage.")

    # --- Main Panel: Display Predictions ---
    st.header("Prediction Results")
    display_predictions()

def train_model(df, value_col, date_col, hour_col, group_col_values, aggregation, selected_numeric_cols=None):
    """Handles the model training process."""
    with st.spinner("Training new model... This may take a moment."):
        try:
            payload = {
                "data": df.to_json(orient='split'),
                "value_column": value_col,
                "date_column": date_col,
                "hour_column": hour_col,
                "group_col_values": group_col_values,
                "selected_numeric_cols": selected_numeric_cols if selected_numeric_cols is not None else [],
                "aggregation": aggregation,
                "architecture": "lstm"
            }
            response = requests.post(f"{API_URL}/train-model", json=payload)
            response.raise_for_status()
            result = response.json()
            st.success(f"Model '{result['model_name']}' trained successfully!")
            # Force rerun to refresh model list and sidebar buttons
            st.rerun()
        except requests.exceptions.RequestException as e:
            error_message = f"Error training model: {e}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    detail = e.response.json().get('detail', e.response.text)
                    error_message = f"Error training model: {detail}"
                except json.JSONDecodeError:
                    error_message = f"Error training model: Status {e.response.status_code} - {e.response.reason}"
            st.error(error_message)

def run_prediction(model_name, t1, t2):
    """Handles running predictions with a selected model and new data."""
    with st.spinner(f"Generating new predictions with model '{model_name}'..."):
        try:
            payload = {
                "model_name": model_name,
                "t1": t1 if t1 is not None else None,
                "t2": t2 if t2 is not None else None
            }
            response = requests.post(f"{API_URL}/predict", json=payload)
            response.raise_for_status()
            st.session_state['predictions'] = response.json()
            st.success("New predictions generated successfully!")
        except requests.exceptions.RequestException as e:
            error_message = f"Error running prediction: {e}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    detail = e.response.json().get('detail', e.response.text)
                    error_message = f"Error running prediction: {detail}"
                except json.JSONDecodeError:
                    error_message = f"Error running prediction: Status {e.response.status_code} - {e.response.reason}"
            st.error(error_message)

def display_predictions():
    """Displays predictions stored in the session state."""
    if 'predictions' in st.session_state:
        predictions_data = st.session_state['predictions']
        model_version = predictions_data.get("model_version")
        predictions = predictions_data.get("predictions")

        if model_version:
            st.info(f"Displaying predictions from model version: **{model_version}**")

        if predictions:
            pred_df = pd.DataFrame(predictions)
            pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])

            # Plot
            fig = px.line(pred_df, x='timestamp', y='predicted_value', title="Hourly Predictions for the Next 10 Days")
            st.plotly_chart(fig, use_container_width=True)

            # Table of predictions
            st.subheader("Prediction Table - Dataframe Streamlit")

            def anomaly_color(row):
                color = ''
                if 'anomaly_status' in row:
                    if row['anomaly_status'] in ['critical', 'red']:
                        color = 'background-color: red'  # light red
                    elif row['anomaly_status'] in ['warning', 'yellow']:
                        color = 'background-color: yellow'  # light yellow
                    elif row['anomaly_status'] in ['maintenance', 'green']:
                        color = 'background-color: green'  # light green
                return [color]*len(row)

            styled_df = pred_df.style.apply(anomaly_color, axis=1)
            st.dataframe(styled_df)

        else:
            st.warning("No prediction data to display for this model.")
            
if __name__ == "__main__":
    main()