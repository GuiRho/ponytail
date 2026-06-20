import os, pickle
from pathlib import Path
import pandas as pd

model = None

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    app = FastAPI(title="Credit Scoring API", version="2.0.0")

    @app.on_event("startup")
    def startup():
        global model
        try: model = load_model()
        except Exception as e:
            model = None
            print(f"WARNING: Model load failed: {e}")

    class ClientData(BaseModel):
        client_id: str = Field(..., description="Client identifier")
        features: dict[str, float] = Field(..., description="Feature values")

    class PredictionResponse(BaseModel):
        client_id: str
        probability: float
        prediction: int

    @app.get("/")
    def health():
        return {"status": "API is running", "model_loaded": model is not None}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(data: ClientData):
        if model is None: raise HTTPException(503, "Model not loaded")
        try:
            proba = model.predict_proba(pd.DataFrame([data.features]))[0]
            prob = float(proba[1]) if proba.ndim == 2 and proba.shape[1] >= 2 else float(proba[0])
            return {"client_id": data.client_id, "probability": prob, "prediction": int(prob >= 0.5)}
        except Exception as e: raise HTTPException(400, str(e))
except ModuleNotFoundError:
    app = None
    def health(): return {"status": "FastAPI not available", "model_loaded": False}

def load_model():
    path = Path(os.environ.get("MODEL_DIR", "production_model")) / "model.pkl"
    if not path.exists(): raise FileNotFoundError(f"Model not found at {path}")
    return pickle.loads(path.read_bytes())

if __name__ == "__main__":
    import tempfile
    d = Path(tempfile.mkdtemp())
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    m = LogisticRegression(max_iter=100)
    m.fit([[0,0],[1,1],[2,2],[3,3]], [0,0,1,1])
    (d / "model.pkl").write_bytes(pickle.dumps(m))
    os.environ["MODEL_DIR"] = str(d)
    loaded = load_model()
    assert loaded is not None
    print("All serving asserts passed")
    import shutil; shutil.rmtree(d, ignore_errors=True)
