from fastapi import FastAPI, Request
from app.model import Detector
import os


app = FastAPI()

MODEL_DIR = os.environ.get("MODEL_DIR", "desklib/ai-text-detector-v1.01")
detector = Detector(MODEL_DIR)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/info")
async def info():
    return {"model_dir": MODEL_DIR}


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    items = data.get("items")

    if items is None:
        single = data.get("item")
        if single is None:
            return {"error": "provide 'items' as list or 'item' as single object"}
        items = [single]

    mode = data.get("mode")
    thresh = data.get("threshold")

    if isinstance(thresh, (int, float)) and isinstance(mode, str):
        return {'error': "chose only one option between 'mode' and 'threshold'"}
    if isinstance(mode, (int, float)):
        threshold = float(mode)
    elif isinstance(mode, str):
        m = mode.lower()
        if m in ("normal", "norm"):
            threshold = 0.5
        elif m in ("strict",):
            threshold = 0.65
        elif m in ("light"):
            threshold = 0.35
        else:
            threshold = 0.5
    elif thresh is not None:
        threshold = float(thresh)
    else:
        threshold = 0.65

    batch_size = data.get("batch_size", 16)
    preds = detector.predict_items(items, batch_size=batch_size, threshold=threshold)
    return {"predictions": preds}
