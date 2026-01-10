from fastapi import FastAPI
from pydantic import BaseModel
from .engine import IDSEngine
import pandas as pd
import uvicorn
import os

app = FastAPI(title="AI IDS Inference API")

# Global Engine Instance (loaded on startup)
engine = None

class PacketData(BaseModel):
    timestamp: float
    src_ip: str
    dst_ip: str
    protocol: str
    length: int
    flags: str

@app.on_event("startup")
def load_artifacts():
    global engine
    # Check for advanced model first
    model_path = "models/artifacts/model_advanced.joblib"
    if not os.path.exists(model_path):
        model_path = "models/artifacts/model.joblib"
    
    print(f"Loading Model from: {model_path}")

    try:
        engine = IDSEngine(
            model_path=model_path,
            preprocessor_path="models/artifacts/preprocessor.joblib",
            extractor_path="models/artifacts/extractor.joblib"
        )
    except Exception as e:
        print(f"Warning: Could not load artifacts: {e}")

@app.post("/predict")
def predict_packet(packet: PacketData):
    if not engine:
        return {"error": "Model not loaded"}
        
    df = pd.DataFrame([packet.dict()])
    result = engine.predict(df)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
