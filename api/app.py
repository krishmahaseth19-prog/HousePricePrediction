import joblib
import pandas as pd

from fastapi import FastAPI

from api.schema import HouseData

app = FastAPI(
    title="House Price Prediction API",
    version="1.0"
)


model = joblib.load(
    "models/house_price_model.joblib"
)


@app.get("/")
def home():

    return {
        "message": "House Price Prediction API Running"
    }


@app.post("/predict")
def predict(data: HouseData):

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