import os
import pickle
import joblib


def save_pickle(model):

    os.makedirs("models", exist_ok=True)

    with open("models/house_price_model.pkl", "wb") as file:
        pickle.dump(model, file)

    print("Pickle model saved")


def save_joblib(model):

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        "models/house_price_model.joblib"
    )

    print("Joblib model saved")