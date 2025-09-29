import pandas as pd
import json
import logging
from typing import List, Optional, Tuple, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def initial_data_processing(
    df: pd.DataFrame, 
    value_col: str, 
    date_col: str, 
    hour_col: Optional[str], 
    group_col_values: dict,
    aggregation: str = 'sum',
    selected_numeric_cols: list = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Processes the initial user-uploaded dataframe.
    - Handles date and optional hour columns.
    - Maps non-numeric columns to integers.
    - Groups data, aggregates the value column, and pivots the data into a wide format suitable for the LSTM model.
    """
    logging.info("Starting initial data processing and pivoting.")

    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()


    # Create timestamp column, handling different input scenarios
    try:
        if hour_col and hour_col in df_copy.columns:
            # If hour_col is present, combine and round to hour
            df_copy['timestamp'] = pd.to_datetime(df_copy[date_col].astype(str) + ' ' + df_copy[hour_col].astype(str))
            df_copy['timestamp'] = df_copy['timestamp'].dt.floor('H')
        elif date_col in df_copy.columns:
            # If date_col is present, check if it's a full datetime or just a date
            df_copy['timestamp'] = pd.to_datetime(df_copy[date_col])
            # If the column has time info, keep as is, else floor to day
            if pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
                if df_copy['timestamp'].dt.hour.nunique() > 1 or df_copy['timestamp'].dt.minute.nunique() > 1:
                    df_copy['timestamp'] = df_copy['timestamp'].dt.floor('H')
                else:
                    df_copy['timestamp'] = df_copy['timestamp'].dt.floor('D')
            else:
                df_copy['timestamp'] = df_copy['timestamp'].dt.floor('D')
        else:
            # Try to find a single datetime column (e.g., 'Time')
            datetime_cols = [col for col in df_copy.columns if pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df_copy[col], errors='coerce'))]
            if len(datetime_cols) == 1:
                df_copy['timestamp'] = pd.to_datetime(df_copy[datetime_cols[0]])
                df_copy['timestamp'] = df_copy['timestamp'].dt.floor('H')
            else:
                raise ValueError("No valid date or datetime column found in the input data.")
    except Exception as e:
        logging.error(f"Error parsing date/time columns: {e}", exc_info=True)
        raise ValueError(f"Could not parse the date and time columns. Please check their format. Error: {e}")

    # Filter rows to only those matching the selected value for each grouping column
    mappings = {}
    grouping_cols = ['timestamp']
    for col, val in group_col_values.items():
        if col in df_copy.columns:
            if not pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy = df_copy[df_copy[col] == val]
                grouping_cols.append(col)
            else:
                # Numeric columns are not used for grouping
                continue
    if aggregation not in ['sum', 'mean']:
        raise ValueError(f"Unsupported aggregation: {aggregation}. Use 'sum' or 'mean'.")
    # Aggregate all numeric columns (including value_col)
    numeric_cols = df_copy.select_dtypes(include='number').columns.tolist()
    # Remove timestamp and grouping cols from aggregation list if present
    for col in grouping_cols:
        if col in numeric_cols:
            numeric_cols.remove(col)
    if not numeric_cols:
        raise ValueError("No numeric columns to aggregate after grouping.")
    processed_df = df_copy.groupby(grouping_cols, as_index=False)[numeric_cols].agg(aggregation)

    # --- Pivoting Logic ---
    group_cols = list(group_col_values.keys())
    if group_cols:
        # If grouping, aggregate and filter to a single group, then output as 'total'
        # Only keep rows matching the selected group values
        for col in group_cols:
            processed_df[col] = processed_df[col].astype(str)
        processed_df['group'] = processed_df[group_cols].agg('_'.join, axis=1)
        # If more than one group, filter to the first (or raise error)
        unique_groups = sorted(processed_df['group'].unique())
        if len(unique_groups) > 1:
            # If user specified a group value, filter to that; else, use the first
            selected_group = None
            group_val = '_'.join(str(group_col_values[c]) for c in group_cols)
            if group_val in unique_groups:
                selected_group = group_val
            else:
                selected_group = unique_groups[0]
            processed_df = processed_df[processed_df['group'] == selected_group]
        # Output as ['timestamp', 'total']
        final_df = processed_df[['timestamp', value_col]].rename(columns={value_col: 'total'})
        mappings['id_to_group'] = {'total': 'total'}
    else:
        # Only include the value_col and any numeric columns picked in the frontend, as-is
        output_cols = ['timestamp']
        if selected_numeric_cols:
            output_cols += [col for col in selected_numeric_cols if col in processed_df.columns]
        if value_col not in output_cols:
            output_cols.append(value_col)
        final_df = processed_df[output_cols]
        # Map column names to themselves for id_to_group
        mappings['id_to_group'] = {col: col for col in output_cols if col != 'timestamp'}

    logging.info("Initial data processing complete. Data has been grouped and pivoted as requested.")
    # Remove header row when saving to CSV in the main pipeline as well
    # (The actual file writing is done outside this function, but document this requirement)
    final_df.attrs['no_header'] = True  # Signal to caller to use header=False
    return final_df, mappings


def process_data_for_lstm(df: pd.DataFrame, time_col: str, value_cols: list, output_csv_path: str, mapping_json_path: str):
    """
    (Legacy Function) Transforms a wide-format DataFrame into a long-format suitable for the LSTM script.
    This function is kept for potential compatibility but is not used in the primary train/predict pipeline.
    """
    logging.info("Transforming data to long format for LSTM processing.")

    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except Exception as e:
        logging.error(f"Could not convert time column '{time_col}' to datetime.", exc_info=True)
        raise ValueError(f"Invalid time column format: {e}") from e

    processed_df = df.melt(
        id_vars=[time_col], value_vars=value_cols, var_name='aplication_name', value_name='total'
    )
    processed_df['date'] = processed_df[time_col].dt.strftime('%Y-%m-%d')
    processed_df['hour'] = processed_df[time_col].dt.hour

    app_names = processed_df['aplication_name'].unique()
    app_name_to_id = {name: i for i, name in enumerate(app_names)}
    id_to_app_name = {i: name for name, i in app_name_to_id.items()}

    processed_df['aplication_name'] = processed_df['aplication_name'].map(app_name_to_id)
    output_df = processed_df[['date', 'hour', 'aplication_name', 'total']]

    try:
        # Always write CSV without header
        output_df.to_csv(output_csv_path, index=False, header=False)
        mappings_content = {
            "time_column_source": time_col,
            "aplication_name": app_name_to_id,
            "id_to_aplication_name": id_to_app_name
        }
        with open(mapping_json_path, 'w') as f:
            json.dump(mappings_content, f, indent=4)
    except Exception as e:
        logging.error("Failed to save processed data or mappings.", exc_info=True)
        raise IOError("File saving failed.") from e
