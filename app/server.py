# app/server.py
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

# ---- Hard-coded config (simple, explicit) ----
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME          = "iris-classifier"
MODEL_VERSION       = "1"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.pyfunc.load_model(MODEL_URI)

# ----- Pydantic schemas with helpful docs + examples -----
class IrisSample(BaseModel):
    sepal_length: float = Field(..., ge=0, description="Sepal length in cm")
    sepal_width:  float = Field(..., ge=0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, description="Petal length in cm")
    petal_width:  float = Field(..., ge=0, description="Petal width in cm")

class PredictRequest(BaseModel):
    samples: List[IrisSample]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "samples": [
                        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                        {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 4.7, "petal_width": 1.5},
                        {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}
                    ]
                }
            ]
        }
    }

# For convenience, return both class ids and human labels
IRIS_LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

class PredictResponse(BaseModel):
    class_id: List[int]    # 0,1,2
    class_label: List[str] # setosa/versicolor/virginica

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"class_id": [0, 1, 2], "class_label": ["setosa", "versicolor", "virginica"]}
            ]
        }
    }

app = FastAPI(
    title="Iris Classifier API",
    description="Predict Iris species from sepal/petal measurements (cm).",
    version="1.0.0",
)

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "model_uri": MODEL_URI}

# @app.post(
#     "/predict",
#     response_model=PredictResponse,
#     tags=["prediction"],
#     summary="Predict Iris species",
#     description="Send one or more Iris samples; returns class id (0,1,2) and label (setosa, versicolor, virginica)."
# )
# def predict(req: PredictRequest) -> PredictResponse:
#     # TODO Run predict
#     return PredictResponse(
#         class_id=[],
#         class_label=[]
#     )
    
@app.post( 
    "/predict", 
    response_model = PredictResponse, 
    tags = ["prediction"], 
    summary = "Predict Iris Species", 
    description = "Send one or more Iris samples; return class id (0, 1, 2) and label (setosa, versicolor, virginica)."
    )
    
def predict(req: PredictRequest) -> PredictResponse: 
    # Convert samples to list of lists (model excpects tabular input) 
    input_data = [
        [s.sepal_length, s.sepal_width, s.petal_length, s.petal_width ]
        for s in req.samples
        ]
    # Run model prediction
    preds = model.predict(input_data) 
    
    # Convert numeric IDs to human-readable labels 
    labels = [IRIS_LABELS[int(i)] for i in preds] 
    
    # Return predictions 
    return PredictResponse( 
        class_id = [int(x) for x in preds], 
        class_label = labels
        ) 


@app.get(
    "/model_info", 
    tags = ["model"], 
    summary = "Retrieve current model version", 
    description = "Retrieve current model version from MlFlow"
)
def get_model_info(): 
    return {
        "model_name": MODEL_NAME, 
        "current_version": MODEL_VERSION, 
        "model_uri": MODEL_URI
        } 

@app.post(
    "/update_model/{version}",
    tags=["model"],
    summary="Update model version",
    description="Choose the model version for inference pipeline"
)
def update_model_version(version: str):
    global model, MODEL_VERSION, MODEL_URI
    MODEL_VERSION = version
    MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    model = mlflow.pyfunc.load_model(MODEL_URI)
    return {"message": f"Model updated to version {version}"}

        
    
# TODO Add endpoint to get the current model serving version
# TODO Add endpoint to update the serving version
# TODO Predict using the correct served version
