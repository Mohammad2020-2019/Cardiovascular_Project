from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Initialize and fit encoders ONCE at startup
le_sex = LabelEncoder().fit(["M", "F"])          # Only M/F allowed
le_cp = LabelEncoder().fit(["TA", "ATA", "NAP", "ASY"])
le_exang = LabelEncoder().fit(["Y", "N"])        # Only Y/N allowed
le_slope = LabelEncoder().fit(["Up", "Flat", "Down"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Heart Prediction Model
class HeartData(BaseModel):
    Age: int
    Sex: str  # 0=Female, 1=Male
    ChestPainType: str  # TA, ATA, NAP, ASY
    Cholesterol: float
    FastingBS: int  # 0=<120 mg/dl, 1=>120 mg/dl
    MaxHR: int
    ExerciseAngina: str  # 0=No, 1=Yes
    Oldpeak: float
    ST_Slope: str  # Up, Flat, Down

# Load Model with error handling
MODEL_PATH = "model/KNeighbors_HeartFailure_Model.joblib"  # Updated path

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.error(f"Model file not found at {os.path.abspath(MODEL_PATH)}")
    raise
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

@app.get("/")
async def root():
    return {"message": "Heart Risk Assessment API"}

@app.post("/Heart_Failure_predict")
async def predict_Heart(data: HeartData):
    try:
        input_data = data.dict()
        input_df = pd.DataFrame([input_data])

        # --- Add Debug Logging ---
        # logger.info("Raw input data: %s", input_data)

        # --- Preprocessing ---
        # Label Encoding (must match training exactly)
        valid_sex = ["M", "F"]
        valid_cp = ["TA", "ATA", "NAP", "ASY"]
        valid_exang = ["Y", "N"]
        valid_slope = ["Up", "Flat", "Down"]

        if input_data["Sex"] not in valid_sex:
            raise ValueError(f"Invalid Sex value. Allowed: {valid_sex}")
        if input_data["ChestPainType"] not in valid_cp:
            raise ValueError(f"Invalid ChestPainType. Allowed: {valid_cp}")
        if input_data["ExerciseAngina"] not in valid_exang:
            raise ValueError(f"Invalid ExerciseAngina. Allowed: {valid_exang}")
        if input_data["ST_Slope"] not in valid_slope:
            raise ValueError(f"Invalid ST_Slope. Allowed: {valid_slope}")

        # Transform categorical features
        input_df["Sex"] = le_sex.transform([input_data["Sex"]])[0]
        input_df["ChestPainType"] = le_cp.transform([input_data["ChestPainType"]])[0]
        input_df["ExerciseAngina"] = le_exang.transform([input_data["ExerciseAngina"]])[0]
        input_df["ST_Slope"] = le_slope.transform([input_data["ST_Slope"]])[0]


        # Select and order EXACTLY the 9 features the model expects
        expected_columns = [
            'Age', 'Sex', 'ChestPainType', 'Cholesterol',
            'FastingBS', 'MaxHR', 'ExerciseAngina',
            'Oldpeak', 'ST_Slope'
        ]

        # Ensure correct column order
        input_processed = input_df[expected_columns]

        # Apply standardization
        numerical_features = ["Age", "Cholesterol", "MaxHR", "Oldpeak"]
        scaler = StandardScaler()
        input_processed [numerical_features] = scaler.fit_transform(input_processed [numerical_features])
        processed_array=input_processed.values

        # --- Add Preprocessed Data Log ---
        #logger.info("Processed features:\n%s",  processed_array)


        # Get prediction
        prediction = model.predict(processed_array)
        prediction_proba = model.predict_proba(processed_array)


        # --- Add Prediction Debug ---
        # logger.info("Model prediction: %s, Probabilities: %s",
        #            prediction, prediction_proba)

        return {
            "diagnosis": "Heart Disease" if prediction[0] == 1 else "No Heart Disease",
            "probability": round(float(np.max(prediction_proba[0])) * 100, 2)
        }

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")