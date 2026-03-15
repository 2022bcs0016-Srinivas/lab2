from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import json

app = FastAPI()

# Load the trained model
model_path = "model.joblib"
model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)

class WineData(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    ph: float
    sulphates: float
    alcohol: float

@app.get("/")
def read_root():
    return {"status": "FastAPI service is running"}

@app.post("/predict")
def predict(data: WineData):
    if model is None:
        return {"error": "Model not loaded"}
    
    # Extract features for prediction
    features = [[
        data.fixed_acidity, data.volatile_acidity, data.citric_acid,
        data.residual_sugar, data.chlorides, data.free_sulfur_dioxide,
        data.total_sulfur_dioxide, data.density, data.ph,
        data.sulphates, data.alcohol
    ]]
    
    prediction = model.predict(features)[0]
    
    return {
        "name": "Srinivas Raghav V C",
        "roll_no": "2022BCS0016",
        "wine_quality": float(prediction)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
