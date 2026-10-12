from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "best_lightgbm_model.pkl")
preprocessor = joblib.load(BASE_DIR / "preprocessor.pkl")

app = FastAPI(title="Home Credit Default Risk API")


class ClientData(BaseModel):
    features: dict[str, object]


@app.get("/")
def root():
    return {"message": "Home Credit Default Risk API is running"}


@app.post("/predict")
def predict(data: ClientData):
    try:
        X = pd.DataFrame([data.features])

        X_processed = preprocessor.transform(X)

        probability = float(model.predict_proba(X_processed)[0, 1])

        return {
            "default_probability": probability
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )
