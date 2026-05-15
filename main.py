
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(title="House Price API")

model = joblib.load("real_estate_model.pkl")



# Request schema

class HouseFeatures(BaseModel):
    longitude: float = Field(gt=-250, lt=-100, description="Longitude")
    latitude: float = Field(gt=28, lt=50, description="Latitude")
    housing_median_age: float = Field(gt=1, lt=60, description="Median Age")
    total_rooms: float = Field(gt=1, lt=50000, description="Total Rooms")
    total_bedrooms: float | None = Field(default=None, gt=1, lt=7000, description="Total Bedrooms")
    population: float = Field(gt=3, lt=50000, description="Population")
    households: float = Field(gt=1, lt=7000, description="Households")
    median_income: float = Field(gt=0, lt=20, description="Median Income")
    ocean_proximity: str = Field(description="Ocean Proximity")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: HouseFeatures):

    df = pd.DataFrame([data.dict()])


    if "total_bedrooms" in df.columns:
        df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

    prediction = model.predict(df)[0]

    return {
        "prediction": float(prediction)
    }