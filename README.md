**Introduction**
This project provides a modular platform for near real-time anomaly detection and predictive maintenance using machine learning techniques. Designed to handle diverse time-series datasets, the system enables users to upload data, train models, and generate forecasts and anomaly alerts through an web application. Core algorithms include LSTM, GRU, ARIMA, autoencoders, and OCSVM, implemented with Python, TensorFlow/Keras, and Streamlit. The platform supports flexible configuration of features and thresholds, interactive visualization of results, and easy management of trained models. By combining machine learning with user-centric design, this project empowers organizations to optimize operational efficiency, minimize downtime, and make data-driven decisions for maintenance planning.

**Structure**
project/
|-- api.py
|-- backend.py
|-- data_preprocessor.py
|-- frontend.py
|-- requirements.txt
|-- setup_guide.md
|-- autoencoder/
| |-- autoencoder_weather.py
|-- db/
| |-- models.db
|-- gru/
| |-- gru_weather.py
|-- logs/
|-- lstm/
| |-- lstm_weather.py
| |-- lstm.py
|-- ocsvm/
| |-- ocsvm_weather.py
|-- original data/
| |-- app01_output.csv
| |-- app02_output.csv
| |-- app03_output.csv
|-- preprocessed_data/
| |-- mappings.json
| |-- output_weather.csv
| |-- ... (other mappings/output files)
|-- synthetic data/
| |-- synthetic_data_app01.csv
| |-- synthetic_generator.py
|-- trained_models/
| |-- model_20250831_190252/
| |-- last_sequence.npy
| |-- model_1.keras
| |-- scaler.save
| |-- source_columns.json

The main files are described below:
• api.py: This file contains the code for the REST API that allows interaction with
the trained models. It uses FastAPI to create endpoints for making predictions and
retrieving model information.
• backend.py: This file contains the backend logic for the application, including
functions for loading models, preprocessing input data, and handling requests from
the API. Uses SQLAlchemy to interact with the SQLite database that stores model
metadata.
• data_preprocessor.py: This file is responsible for preprocessing the raw data,
including handling missing values, normalizing features, and splitting the data into
training and testing sets.
• frontend.py: This file contains the code for the user interface, allowing users to
interact with the system through a web-based interface. It uses Streamlit to create
an interactive and user-friendly experience.
• requirements.txt: This file lists all the Python dependencies required to run the
project, making it easy to set up the environment using pip.
• setup_guide.md: This file provides instructions on how to set up and run the
project, including environment setup and usage guidelines.
• autoencoder: This directory contains code specific to the Autoencoder model
implementation. Only contains the python file used for testing with the weather
dataset.
• db/: This directory contains the database file used to store model metadata and
other relevant information.
• gru/: This directory contains code specific to the GRU model implementation. Only
contains the python file used for testing with the weather dataset.
• logs/: This directory is used to store log files generated during model training and
evaluation.
• lstm/: This directory contains code specific to the LSTM model implementation.
Contains both the python file used for testing with the weather dataset and the
LSTM model used in the project.
• ocsvm/: This directory contains code specific to the OCSVM implementation. Only
contains the python file used for testing with the weather dataset.
• original_data/: This directory contains the original datasets used for training and
evaluation. Contains the synthectic datasets generated for the applications.
• preprocessed_data/: This directory contains scripts and files related to data
preprocessing, including mappings and preprocessed output files. Contains the pre-
processed data results from data_preprocessor.py and the output_weather.csv and
mappings.json files used in the testing of the models.
• synthetic_data/: This directory contains scripts for generating synthetic data
and the resulting synthetic datasets. Contains the synthetic dataset generated for
application 1 and the script used to generate it.
• trained_models/: This directory stores trained models along with their associated
files such as scalers and source column mappings.


**Installing the requirements**
Before starting the application, the user needs to install the required dependencies listed
in the requirements.txt file. This can be done using pip:

pip install r requirements.txt

**Running the application**
After successful installation of the requirements, to start the application, the user needs to
run both the api.py and the frontend.py file in seperate command line windows. This
will launch a Streamlit web application where the user can interact with the system.

python api.py
streamlit run frontend.py

