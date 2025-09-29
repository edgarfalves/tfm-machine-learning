from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- Local Imports ---
try:
    from backend import (
        train_model_pipeline,
        run_prediction_pipeline,
        get_models_by_type,
        delete_model_by_name,
        get_database_stats,
        get_predictions_by_model_name
    )
except ImportError as e:
    raise RuntimeError(f"Failed to import from backend: {e}. Please ensure backend.py is in the correct path.") from e

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- FastAPI Application ---
app = FastAPI(
    title="Prediction Pipeline API",
    description="An API to run and manage time-series prediction models.",
    version="3.0"
)

# --- Pydantic Models for Request/Response ---

class TrainPayload(BaseModel):
    data: str = Field(..., description="A JSON string of the dataframe, created with pandas `to_json(orient='split')`")
    value_column: str
    date_column: str
    hour_column: Optional[str] = None
    group_col_values: dict = Field(default_factory=dict, description="Dict of {col: value} for grouping. Only rows matching these values are kept.")
    selected_numeric_cols: list = Field(default_factory=list, description="List of numeric columns selected in the frontend to include in the output.")
    aggregation: str = Field('sum', description="Aggregation method for value column: 'sum' or 'mean'")
    architecture: str = Field("lstm", example="lstm")

class PredictionPayload(BaseModel):
    model_name: str
    t1: Optional[float] = Field(None, nullable=True)
    t2: Optional[float] = Field(None, nullable=True)

class PipelineResponse(BaseModel):
    message: str
    model_name: str
    model_id: int

class ModelInfo(BaseModel):
    id: int
    model_type: str
    model_name: str
    created_at: str
    extra_info: Dict[str, Any]

class PredictionInfo(BaseModel):
    id: int
    model_id: int
    timestamp: str
    predicted_value: float
    source_columns: str
    anomaly_status: str = None

class DatabaseStats(BaseModel):
    total_models: int
    total_predictions: int

class LatestPredictionsResponse(BaseModel):
    model_version: Optional[str] = None
    predictions: List[Dict[str, Any]]


# --- API Endpoints ---

@app.post("/train-model", response_model=PipelineResponse)
def train_model_endpoint(payload: TrainPayload):
    logger.info(f"Received request to train model with architecture: '{payload.architecture}'")
    try:
        result = train_model_pipeline(
            data_json=payload.data,
            value_column=payload.value_column,
            date_column=payload.date_column,
            hour_column=payload.hour_column,
            group_col_values=payload.group_col_values,
            selected_numeric_cols=payload.selected_numeric_cols,
            aggregation=payload.aggregation,
            architecture=payload.architecture
        )
        logger.info(f"Training pipeline for model '{result['model_name']}' completed successfully.")
        return result
    except ValueError as ve:
        logger.warning(f"Validation error in training request: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while training the model.")


# --- Predict endpoint: Only requires model_name ---
@app.post("/predict", response_model=LatestPredictionsResponse)
async def predict(payload: PredictionPayload):
    """
    Generate new predictions for a given model name, with optional anomaly thresholds t1 and t2.
    """
    try:
        logger.info(f"Received request for prediction with model: '{payload.model_name}'")
        # Call backend logic to generate new predictions for the model, passing t1 and t2
        result = run_prediction_pipeline(payload.model_name, t1=payload.t1, t2=payload.t2)
        return result
    except Exception as e:
        logger.error(f"Prediction failed for model '{payload.model_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/models/{model_type}", response_model=List[ModelInfo])
def list_models_by_type_endpoint(model_type: str):
    logger.info(f"Request received for models of type: {model_type}")
    try:
        models = get_models_by_type(model_type)
        return models
    except Exception as e:
        logger.error(f"Failed to fetch models for type '{model_type}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve models from the database.")

@app.delete("/models/{model_name}", response_model=Dict[str, str])
def delete_model_endpoint(model_name: str):
    logger.info(f"Request to delete model: {model_name}")
    try:
        result = delete_model_by_name(model_name)
        return result
    except Exception as e:
        logger.error(f"Failed to delete model '{model_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {e}")

@app.get("/database-stats", response_model=DatabaseStats)
def get_stats_endpoint():
    logger.info("Request received for database stats.")
    try:
        stats = get_database_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to fetch database stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve database statistics.")

@app.get("/predictions/{model_name}", response_model=List[PredictionInfo])
def get_predictions_by_model_name_endpoint(model_name: str):
    """
        Retrieves predictions for a given model name from the database.
    """
    logger.info(f"Request received for predictions from model: {model_name}")
    try:
        predictions = get_predictions_by_model_name(model_name)
        if not predictions:
            logger.warning(f"No predictions found for model: {model_name}")
        return predictions
    except Exception as e:
        logger.error(f"Failed to fetch predictions for model '{model_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve predictions.")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
