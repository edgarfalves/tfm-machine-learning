import os
import json
import sys
from datetime import datetime
import pandas as pd
import subprocess
import logging
import shutil
from typing import Optional

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

try:
    from data_preprocessor import initial_data_processing, process_data_for_lstm
except ImportError:
    def initial_data_processing(*args, **kwargs):
        raise ImportError("Could not import 'initial_data_processing'. Make sure data_preprocessor.py is accessible.")
    def process_data_for_lstm(*args, **kwargs):
        raise ImportError("Could not import 'process_data_for_lstm'. Make sure data_preprocessor.py is accessible.")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "project/db", "models.db")
LOG_DIR = os.path.join(BASE_DIR, "project/logs")
DATA_DIR = os.path.join(BASE_DIR, "project/preprocessed_data")
LSTM_SCRIPT_PATH = os.path.join(BASE_DIR, "project/lstm", "lstm.py")
MODELS_DIR = os.path.join(BASE_DIR, "project", "trained_models")


os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Base = declarative_base()
Session = sessionmaker(bind=engine)

class PredictionModel(Base):
    __tablename__ = 'prediction_models'
    id = Column(Integer, primary_key=True)
    model_type = Column(String(50), nullable=False)
    model_name = Column(String(255), nullable=False, unique=True)
    model_path = Column(String(500), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    extra_info = Column(Text, nullable=True)
    predictions = relationship("Prediction", back_populates="model", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('prediction_models.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    predicted_value = Column(Float, nullable=False)
    source_columns = Column(Text, nullable=True)
    anomaly_status = Column(String(20), nullable=True)  # New column for anomaly status
    created_at = Column(DateTime, default=datetime.utcnow)
    model = relationship("PredictionModel", back_populates="predictions")

Base.metadata.create_all(engine)

def train_model_pipeline(data_json: str, value_column: str, date_column: str, hour_column: Optional[str], group_col_values: dict, selected_numeric_cols: list, aggregation: str, architecture: str):
    if architecture.lower() != 'lstm':
        raise ValueError(f"Unsupported architecture: {architecture}. Only 'lstm' is supported.")
    
    session = Session()
    model_id = None
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"model_{run_timestamp}"
    
    try:
        logging.info(f"Registering new model '{model_name}'...")
        extra_info = {"columns_processed": list(group_col_values.keys()), "training_timestamp": datetime.now().isoformat(), "status": "pending"}
        model_path = os.path.join(MODELS_DIR, model_name)
        new_model = PredictionModel(
            model_type=architecture, model_name=model_name,
            model_path=model_path, extra_info=json.dumps(extra_info)
        )
        session.add(new_model)
        session.commit()
        model_id = new_model.id
        logging.info(f"Successfully registered model with ID: {model_id}")

        df = pd.read_json(data_json, orient='split')
        processed_df, mappings = initial_data_processing(df, value_column, date_column, hour_column, group_col_values, aggregation, selected_numeric_cols)
        # Ensure value_column is always present in mappings for downstream use
        mappings["value_column"] = value_column
        
        processed_csv_path = os.path.join(DATA_DIR, f"output_{run_timestamp}.csv")
        mappings_path = os.path.join(DATA_DIR, f"mappings_{run_timestamp}.json")
        
        processed_df.to_csv(processed_csv_path, index=False)
        with open(mappings_path, 'w') as f:
            json.dump(mappings, f)

        logging.info(f"Executing LSTM training for model ID {model_id}...")
        result = subprocess.run(
            ["python", LSTM_SCRIPT_PATH, "train", processed_csv_path, mappings_path, str(model_id), model_path],
            capture_output=True, text=True, check=False, encoding='utf-8'
        )

        if result.returncode != 0:
            raise Exception(f"LSTM training failed: {result.stderr}")

        logging.info("LSTM training executed successfully.")

        model = session.query(PredictionModel).filter_by(id=model_id).first()
        if model:
            current_extra_info = json.loads(model.extra_info)
            current_extra_info['status'] = 'completed'
            current_extra_info['completed_at'] = datetime.now().isoformat()
            model.extra_info = json.dumps(current_extra_info)
            session.commit()
            logging.info(f"Updated model '{model_name}' status to 'completed'.")
        
        return {"message": "Training pipeline executed successfully!", "model_name": model_name, "model_id": model_id}

    except Exception as e:
        logging.error(f"An error occurred in the training pipeline for model '{model_name}': {e}", exc_info=True)
        if session.is_active: session.rollback()
        if model_id:
            try:
                model = session.query(PredictionModel).filter_by(id=model_id).first()
                if model:
                    info = json.loads(model.extra_info) if model.extra_info and model.extra_info.startswith('{') else {}
                    info.update({'status': 'failed', 'error': str(e)})
                    model.extra_info = json.dumps(info)
                    session.commit()
            except Exception as db_err:
                logging.error(f"Could not update model status to failed: {db_err}")
        raise e
    finally:
        if session.is_active: session.close()

def run_prediction_pipeline(model_name: str, t1: float = None, t2: float = None):
    """
    Generate new predictions for a given model name, with optional anomaly thresholds t1 and t2.
    """
    session = Session()
    try:
        model = session.query(PredictionModel).filter_by(model_name=model_name).first()
        if not model:
            raise ValueError(f"Model '{model_name}' not found.")

        # Build command for LSTM script, passing t1 and t2 if provided
        cmd = ["python", LSTM_SCRIPT_PATH, "predict", str(model.id), model.model_path]
        if t1 is not None:
            cmd += ["--t1", str(t1)]
        if t2 is not None:
            cmd += ["--t2", str(t2)]

        result = subprocess.run(
            cmd,
            capture_output=True, text=True, check=False, encoding='utf-8'
        )
        if result.stdout:
            print("[LSTM subprocess stdout]:\n", result.stdout)
        if result.stderr:
            print("[LSTM subprocess stderr]:\n", result.stderr)
        if result.returncode != 0:
            raise Exception(f"Prediction failed: {result.stderr}")

        try:
            predictions_list = json.loads(result.stdout)
        except Exception as e:
            logging.error(f"Failed to parse predictions JSON from LSTM script: {e}\nOutput was: {result.stdout}")
            raise

        # Pass through predictions as received from LSTM script
        save_predictions_to_db(predictions_list, model.id)
        predictions = get_predictions_by_model_name(model_name)
        logging.info(f"Predictions for model '{model_name}' generated successfully.")
        return {"model_version": model_name, "predictions": predictions}
    finally:
        session.close()


def save_predictions_to_db(predictions: list, model_id: int):
    if not predictions:
        logging.warning(f"save_predictions_to_db called with empty predictions list for model_id {model_id}.")
        return
    logging.info(f"save_predictions_to_db called with {len(predictions)} predictions for model_id {model_id}.")
    logging.debug(f"First prediction sample: {predictions[0] if predictions else 'N/A'}")
    session = Session()
    try:
        # Clear old predictions for this model
        session.query(Prediction).filter_by(model_id=model_id).delete()
        for idx, p in enumerate(predictions):
            try:
                src_cols = p.get('source_columns')
                if isinstance(src_cols, dict):
                    src_cols = json.dumps(src_cols)
                anomaly_status = p.get('anomaly_status')
                prediction_entry = Prediction(
                    model_id=model_id,
                    timestamp=pd.to_datetime(p['timestamp']).to_pydatetime(),
                    predicted_value=float(p['predicted_value']),
                    source_columns=src_cols,
                    anomaly_status=anomaly_status
                )
                session.add(prediction_entry)
            except Exception as pred_ex:
                logging.error(f"Error processing prediction #{idx}: {p} | Exception: {pred_ex}")
        session.commit()
        logging.info(f"Successfully saved {len(predictions)} predictions for model_id {model_id}.")
    except Exception as e:
        session.rollback()
        logging.error(f"Error saving predictions to database: {e}", exc_info=True)
        raise
    finally:
        session.close()

def get_models_by_type(model_type: str):
    session = Session()
    try:
        models = session.query(PredictionModel).filter_by(model_type=model_type.lower()).order_by(PredictionModel.created_at.desc()).all()
        return [{'id': m.id, 'model_type': m.model_type, 'model_name': m.model_name,
                 'created_at': m.created_at.isoformat(), 'extra_info': json.loads(m.extra_info) if m.extra_info else {}}
                for m in models]
    finally:
        session.close()

def delete_model_by_name(model_name: str):
    session = Session()
    try:
        model = session.query(PredictionModel).filter_by(model_name=model_name).first()
        if not model:
            raise ValueError(f"Model '{model_name}' not found.")
        
        # Delete model files
        if os.path.exists(model.model_path):
            shutil.rmtree(model.model_path)
            
        session.delete(model)
        session.commit()
        return {"message": f"Model '{model_name}' and its predictions have been deleted."}
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_database_stats():
    session = Session()
    try:
        return {"total_models": session.query(PredictionModel).count(), "total_predictions": session.query(Prediction).count()}
    finally:
        session.close()

def _format_prediction(p):
    """Helper to format a prediction object into a dictionary."""
    return {
        'id': p.id,
        'model_id': p.model_id,
        'timestamp': p.timestamp.isoformat(),
        'predicted_value': p.predicted_value,
        'source_columns': json.loads(p.source_columns) if p.source_columns and p.source_columns.startswith('{') else p.source_columns,
        'anomaly_status': p.anomaly_status
    }

def get_all_predictions(skip: int = 0, limit: int = 100):
    session = Session()
    try:
        predictions = session.query(Prediction).order_by(Prediction.timestamp.desc()).offset(skip).limit(limit).all()
        return [_format_prediction(p) for p in predictions]
    finally:
        session.close()

def get_predictions_by_model_name(model_name: str):
    """Get all predictions for a given model name."""
    session = Session()
    try:
        model = session.query(PredictionModel).filter_by(model_name=model_name).first()
        if not model:
            return []
        return [_format_prediction(p) for p in model.predictions]
    finally:
        session.close()


def get_latest_predictions():
    session = Session()
    try:
        latest_model = session.query(PredictionModel).filter(PredictionModel.extra_info.like('%"status": "completed"%'))\
            .order_by(PredictionModel.created_at.desc()).first()
        if not latest_model: return {"model_version": None, "predictions": []}
        
        predictions = session.query(Prediction).filter_by(model_id=latest_model.id).all()
        
        return {"model_version": latest_model.model_name,
                "predictions": [{'timestamp': p.timestamp.isoformat(), 'application_name': p.application_name,
                                 'predicted_value': p.predicted_value} for p in predictions]}
    finally:
        session.close()
