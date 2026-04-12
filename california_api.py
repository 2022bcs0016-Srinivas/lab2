# Lab 8: California Housing Inference Service
# Student: Srinivas Raghav V C
# Roll No: 2022BCS0016

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title='California Housing Prediction API',
    description='ML Inference Service - Lab 8',
    version='1.0.0',
)

ROLL_NO = '2022BCS0016'
STUDENT_NAME = 'Srinivas Raghav V C'

model = None
scaler = None


class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


class HousingPredictionResponse(BaseModel):
    name: str
    roll_no: str
    predicted_house_value: float


@app.on_event('startup')
async def load_model():
    global model, scaler
    try:
        model = joblib.load('app/artifacts/california_model.joblib')
        scaler = joblib.load('app/artifacts/california_scaler.joblib')
        print(f'{ROLL_NO} - California model loaded successfully')
    except Exception as exc:
        print(f'Error loading California model artifacts: {exc}')
        model = None
        scaler = None


@app.get('/health')
async def health():
    return {
        'status': 'healthy',
        'roll_no': ROLL_NO,
        'student': STUDENT_NAME,
        'service': 'california-housing-api',
    }


@app.post('/predict', response_model=HousingPredictionResponse)
async def predict(features: HousingFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail='California housing model is not loaded')

    try:
        feature_values = [
            features.MedInc,
            features.HouseAge,
            features.AveRooms,
            features.AveBedrms,
            features.Population,
            features.AveOccup,
            features.Latitude,
            features.Longitude,
        ]

        X = np.array([feature_values], dtype=float)
        X_scaled = scaler.transform(X)
        prediction = float(model.predict(X_scaled)[0])

        return HousingPredictionResponse(
            name=STUDENT_NAME,
            roll_no=ROLL_NO,
            predicted_house_value=round(prediction, 4),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
