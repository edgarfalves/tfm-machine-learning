pip install -r requirements.txt
python api.pystreamlit run frontend.py
## Introduction

This project provides a modular platform for near real-time anomaly detection and predictive maintenance using machine learning techniques. Designed to handle diverse time-series datasets, the system enables users to upload data, train models, and generate forecasts and anomaly alerts through a web application.

Core algorithms include **LSTM**, **GRU**, **ARIMA**, **autoencoders**, and **OCSVM**, implemented with Python, TensorFlow/Keras, and Streamlit. The platform supports flexible configuration of features and thresholds, interactive visualization of results, and easy management of trained models. By combining machine learning with user-centric design, this project empowers organizations to optimize operational efficiency, minimize downtime, and make data-driven decisions for maintenance planning.

## Structure

```
project/
|-- api.py
|-- backend.py
|-- data_preprocessor.py
|-- frontend.py
|-- requirements.txt
|-- setup_guide.md
|-- autoencoder/
|   |-- autoencoder_weather.py
|-- db/
|   |-- models.db
|-- gru/
|   |-- gru_weather.py
|-- logs/
|-- lstm/
|   |-- lstm_weather.py
|   |-- lstm.py
|-- ocsvm/
|   |-- ocsvm_weather.py
|-- original_data/
|   |-- app01_output.csv
|   |-- app02_output.csv
|   |-- app03_output.csv
|-- preprocessed_data/
|   |-- mappings.json
|   |-- output_weather.csv
|   |-- ... (other mappings/output files)
|-- synthetic_data/
|   |-- synthetic_data_app01.csv
|   |-- synthetic_generator.py
|-- trained_models/
|   |-- model_20250831_190252/
|       |-- last_sequence.npy
|       |-- model_1.keras
|       |-- scaler.save
|       |-- source_columns.json
```

## Main Files

- **api.py**: REST API for interacting with trained models, using FastAPI for predictions and model info.
- **backend.py**: Backend logic, including model loading, data preprocessing, and API request handling. Uses SQLAlchemy for SQLite database management.
- **data_preprocessor.py**: Preprocesses raw data (handles missing values, normalization, train/test split).
- **frontend.py**: Web-based user interface built with Streamlit.
- **requirements.txt**: Lists Python dependencies for easy setup.
- **setup_guide.md**: Instructions for environment setup and usage.
- **autoencoder/**: Autoencoder model code for weather dataset.
- **db/**: Database file for model metadata and related info.
- **gru/**: GRU model code for weather dataset.
- **logs/**: Stores log files from training and evaluation.
- **lstm/**: LSTM model code and scripts.
- **ocsvm/**: OCSVM model code for weather dataset.
- **original_data/**: Original and synthetic datasets for training/evaluation.
- **preprocessed_data/**: Data preprocessing scripts and output files.
- **synthetic_data/**: Synthetic data generation scripts and datasets.
- **trained_models/**: Trained models and associated files (scalers, mappings).

## Installing the Requirements

Before starting the application, install the required dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Running the Application

After installing the requirements, start the application by running both `api.py` and `frontend.py` in separate command line windows:

```bash
python api.py
streamlit run frontend.py
```

## Features

- Near real-time anomaly detection
- Predictive maintenance for time-series data
- Multiple ML models: LSTM, GRU, ARIMA, Autoencoder, OCSVM
- Interactive web interface (Streamlit)
- REST API for model predictions
- Flexible configuration of features and thresholds
- Visualization of results and anomaly alerts
- Easy management of trained models

## Usage Examples

### API Request Example
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"data": [ ... ]}'
```

### Streamlit UI
Open your browser at `http://localhost:8501` after running the frontend to interact with the dashboard.


## Known Issues / TODO

- Add more model types (e.g., ARIMA integration)
- Improve UI for configuration
- Add more usage examples and documentation
- Performance optimizations


## License

This project is licensed under the terms of the LICENSE file in this repository.