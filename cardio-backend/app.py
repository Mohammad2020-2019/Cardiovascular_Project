from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from openai import OpenAI
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
from dotenv import load_dotenv
import traceback
import re
import pickle

# Load environment variables
load_dotenv()

# Initialize app and logging
app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATASET_URL = "https://e-react-node-backend-22ed6864d5f3.herokuapp.com/table/patients_analysis"

# Load model and scalers
try:
    model = joblib.load("model/KNeighbors_HeartFailure_Model.joblib")
    with open("model/standard_scaler.pkl", "rb") as f:
        ss = pickle.load(f)
    with open("model/min_max_scaler.pkl", "rb") as f:
        mms = pickle.load(f)
    logger.info("All model assets loaded successfully")
except Exception as e:
    logger.critical(f"Failed to load model assets: {str(e)}")
    raise

# Initialize encoders
le_sex = LabelEncoder().fit(["M", "F"])
le_cp = LabelEncoder().fit(["TA", "ATA", "NAP", "ASY"])
le_exang = LabelEncoder().fit(["Y", "N"])
le_slope = LabelEncoder().fit(["Up", "Flat", "Down"])

# Feature configuration
SELECTED_FEATURES = ['Age', 'Sex', 'ChestPainType', 'Cholesterol',
                     'FastingBS', 'MaxHR', 'ExerciseAngina', 'Oldpeak',
                     'ST_Slope']
NUMERICAL_FEATURES = ['Age', 'Cholesterol', 'MaxHR', 'Oldpeak']

class ChatRequest(BaseModel):
    message: str

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Field mapping constants
CHEST_PAIN_MAP = {0: "TA", 1: "ATA", 2: "NAP", 3: "ASY"}
SLOPE_MAP = {0: "Up", 1: "Flat", 2: "Down"}

def preprocess_data(patient_data: dict) -> np.ndarray:
    """Preprocess patient data for model prediction"""
    try:
        # Field mapping and validation
        processed = {
            "Age": float(patient_data["age"]),
            "Sex": "M" if patient_data.get("gender") == "Male" else "F",
            "ChestPainType": CHEST_PAIN_MAP.get(patient_data.get("chestpain", 3), "ASY"),
            "Cholesterol": float(patient_data["serum_cholesterol"]),
            "FastingBS": patient_data.get("fastingbloodsugar", 0),
            "MaxHR": float(patient_data["maxheartrate"]),
            "ExerciseAngina": "Y" if patient_data.get("exerciseangia", 0) == 1 else "N",
            "Oldpeak": float(patient_data.get("oldpeak", 0.0)),
            "ST_Slope": SLOPE_MAP.get(patient_data.get("slope", 1), "Flat")
        }

        # Validate required fields
        required_fields = ["age", "serum_cholesterol", "maxheartrate"]
        for field in required_fields:
            if field not in patient_data:
                raise ValueError(f"Missing required field: {field}")

        # Encode categorical features
        processed["Sex"] = le_sex.transform([processed["Sex"]])[0]
        processed["ChestPainType"] = le_cp.transform([processed["ChestPainType"]])[0]
        processed["ExerciseAngina"] = le_exang.transform([processed["ExerciseAngina"]])[0]
        processed["ST_Slope"] = le_slope.transform([processed["ST_Slope"]])[0]

        # Create DataFrame
        df = pd.DataFrame([processed], columns=SELECTED_FEATURES)
        logger.info(f"Extracted data \n")
        logger.info(df)
        # Apply feature scaling
        df[['Oldpeak']] = mms.transform(df[['Oldpeak']])
        df[NUMERICAL_FEATURES[:-1]] = ss.transform(df[NUMERICAL_FEATURES[:-1]])

        # Validate feature structure
        if list(df.columns) != SELECTED_FEATURES:
            raise ValueError(f"Feature mismatch. Expected: {SELECTED_FEATURES}, Got: {list(df.columns)}")

        return df.values

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

#  patient matching
def fetch_patient_data(identifier: str):
    try:
        logger.info(f"Fetching data from: {DATASET_URL}")
        response = requests.get(DATASET_URL)
        response.raise_for_status()
        patients = response.json()

        # Log raw API response
        logger.info(f"Raw API response sample: {patients[:1]}")  # Log first patient
        logger.info(f"Total patients fetched: {len(patients)}")

        # Check API response structure
        if not patients or not isinstance(patients, list):
            raise ValueError("Invalid API response structure")

        # Numeric ID search
        clean_id = re.sub(r'\D', '', identifier)
        if clean_id:
            logger.info(f"Searching by ID: {clean_id}")
            for patient in patients:
                if any(str(patient.get(key)) == clean_id
                       for key in ['id', 'patient_id']):
                    logger.info("Found patient by ID:")
                   # logger.info(patient)  # Log raw patient data
                    return patient

        # Name search with partial matching
        name_query = re.sub(r'[^a-zA-Z ]', '', identifier).lower().strip()
        if name_query:
            logger.info(f"Searching by name: {name_query}")
            for patient in patients:
                fname = patient.get('FName', '').lower()
                lname = patient.get('LName', '').lower()
                if (name_query in f"{fname} {lname}" or
                        name_query in fname or
                        name_query in lname):
                    logger.info("Found patient by name:")
                   # logger.info(patient)  # Log raw patient data
                    return patient

        logger.warning("No patient found matching criteria")
        return None
    except Exception as e:
        logger.error(f"Fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail="Data fetch failed")


def extract_identifier(user_message: str) -> str:
    """Improved identifier extraction with better regex"""
    try:
        # First try regex pattern
        patterns = [
            r'(?:patient|id|pat)\s*[:]?\s*(\d+)',  # Match IDs like "Patient ID: 123"
            r'\b\d+\b',                            # Standalone numbers
            r'\b[A-Za-z]+\s[A-Za-z]+\b'            # Full names
        ]

        for pattern in patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                return match.group(1) if pattern == patterns[0] else match.group(0)

        # If no match, use OpenAI
        return get_openai_identifier(user_message)  # Your existing OpenAI extraction

    except Exception as e:
        logger.error(f"Identifier extraction failed: {str(e)}")
        return "not found"

def get_openai_identifier(message: str):
    """Separated OpenAI extraction"""
    try:
        extraction_prompt = f"""Extract patient identifier from:
        '{message}'. Return ONLY the identifier or 'not found'"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": extraction_prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI extraction failed: {str(e)}")
        return "not found"



# Add this error handler
@app.exception_handler(404)
async def not_found_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": "Endpoint not found"},
    )

@app.options("/chat")
async def options_handler():
    return JSONResponse(content={}, status_code=200)

# Update the chat handler endpoint
@app.post("/chat")
async def chat_handler(request: ChatRequest):
    try:
        identifier = extract_identifier(request.message.strip())
        if identifier == "not found":
            raise HTTPException(status_code=400, detail="No valid identifier found in message")

        logger.info(f"Searching for patient: {identifier}")
        patient = fetch_patient_data(identifier)

        if not patient:
            logger.warning(f"Patient not found: {identifier}")
            raise HTTPException(status_code=404, detail="Patient not found")

        # Log complete patient data
        #logger.info("Full patient data retrieved from web:")
        #logger.info(patient)

        # Preprocessing and prediction
        try:
            processed_data = preprocess_data(patient)
            logger.debug(f"Processed data shape: {processed_data.shape}")

            #logger.info(processed_data)
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=422, detail=f"Data validation error: {str(e)}")

        try:
            prediction = model.predict(processed_data)
            probability = model.predict_proba(processed_data)[0][1] * 100
            logger.info(f"Prediction result: {prediction[0]} ({probability:.1f}%)")
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail="Model prediction error")

        return {
            "status": "success",
            "patient": {
                "id": patient.get('patient_id', patient.get('id')),
                "name": f"{patient.get('FName', '')} {patient.get('LName', '')}",
                "analysis": {
                    "risk_level": "Heart failure risk" if prediction[0] == 1 else "No heart failure risk",
                    "probability": round(probability, 1)
                }
            }
        }

    except HTTPException as he:
        raise
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Analysis failed")

    # Temporary test route to verify data processing
@app.post("/test_patient/{patient_id}")
async def test_patient(patient_id: str):
    patient = fetch_patient_data(patient_id)
    if not patient:
        return {"error": "Patient not found"}

    try:
        processed = preprocess_data(patient)
        prediction = model.predict(processed)
        return {
            "patient_data": patient,
            "processed_features": processed.tolist(),
            "prediction": int(prediction[0])
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
@app.get("/features")
async def show_features():
    return {
        "selected_features": selected_features,
        "model_features": model.feature_names_in_.tolist()
    }
@app.get("/")
async def root():
    return {"message": "Heart Risk Assessment API"}

@app.get("/test_env")
async def test_env():
    return {
        "openai_key_set": os.getenv("OPENAI_API_KEY") is not None,
        "backend_port": os.getenv("PORT")
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)