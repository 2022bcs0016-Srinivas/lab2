# Lab 4: FastAPI Inference Service
# Student: Srinivas Raghav V C
# Roll No: 2022BCS0016

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(
    title="Wine Quality Prediction API",
    description="ML Inference Service - Lab 4",
    version="1.0.0"
)

# Roll number identifier
ROLL_NO = "2022BCS0016"
STUDENT_NAME = "Srinivas Raghav V C"

# Load model and scaler
model = None
scaler = None

class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

class PredictionResponse(BaseModel):
    name: str
    roll_no: str
    wine_quality: float

@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        model = joblib.load('app/artifacts/model.joblib')
        scaler = joblib.load('app/artifacts/scaler.joblib')
        print(f"{ROLL_NO} - Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Create dummy model for testing
        model = None
        scaler = None

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "roll_no": ROLL_NO,
        "student": STUDENT_NAME
    }

@app.get("/")
async def root():
    return {
        "message": "Wine Quality Prediction API",
        "roll_no": ROLL_NO,
        "student": STUDENT_NAME,
        "endpoints": ["/predict", "/health"]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: WineFeatures):
    try:
        # Prepare features
        feature_values = [
            features.fixed_acidity,
            features.volatile_acidity,
            features.citric_acid,
            features.residual_sugar,
            features.chlorides,
            features.free_sulfur_dioxide,
            features.total_sulfur_dioxide,
            features.density,
            features.pH,
            features.sulphates,
            features.alcohol
        ]
        
        X = np.array([feature_values])
        
        # Scale features
        if scaler is not None:
            X = scaler.transform(X)
        
        # Predict
        if model is not None:
            prediction = model.predict(X)[0]
        else:
            # Fallback for testing
            prediction = 5.0
        
        return PredictionResponse(
            name=STUDENT_NAME,
            roll_no=ROLL_NO,
            wine_quality=round(float(prediction), 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
