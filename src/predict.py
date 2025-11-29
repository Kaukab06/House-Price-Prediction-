# src/predict.py
import joblib
import pandas as pd

model = joblib.load("house_price_model.joblib")

def predict_house(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return float(pred)

if __name__ == "__main__":
    sample = {
        "MedInc": 5.0,
        "HouseAge": 20,
        "AveRooms": 4,
        "AveBedrms": 1,
        "Population": 800,
        "AveOccup": 3,
        "Latitude": 35,
        "Longitude": -120
    }
    print("Prediction:", predict_house(sample))
