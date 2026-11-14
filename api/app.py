import os
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download

from api.schema import HouseData


app = FastAPI(
    title="House Price Prediction API",
    version="1.0",
    description="Machine Learning-Based House Price Prediction API"
)


# Hugging Face model configuration
MODEL_REPO_ID = "Krishmahaseth/house-price-prediction-model"
MODEL_FILENAME = "house_price_model.joblib"


# Download the model from Hugging Face
model_path = hf_hub_download(
    repo_id=MODEL_REPO_ID,
    filename=MODEL_FILENAME,
    repo_type="model"
)


# Load the trained machine learning model
model = joblib.load(model_path)


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API Running",
        "model": "Extra Trees Regressor",
        "status": "Ready"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(data: HouseData):

    try:
        input_df = pd.DataFrame([{
            "bedrooms": data.bedrooms,
            "bathrooms": data.bathrooms,
            "sqft_living": data.sqft_living,
            "sqft_lot": data.sqft_lot,
            "floors": data.floors,
            "waterfront": data.waterfront,
            "view": data.view,
            "condition": data.condition,
            "sqft_above": data.sqft_above,
            "sqft_basement": data.sqft_basement,
            "yr_built": data.yr_built,
            "yr_renovated": data.yr_renovated,
            "city": data.city,
            "statezip": data.statezip,
            "country": data.country,
            "year": data.year,
            "month": data.month
        }])

        prediction = model.predict(input_df)

        return {
            "predicted_price": float(prediction[0])
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )