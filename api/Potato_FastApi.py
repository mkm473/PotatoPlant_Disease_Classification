from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import httpx
import os

app = FastAPI()
ENDPOINT_URL = os.getenv("DOWNSTREAM_ENDPOINT")

MODEL = tf.keras.models.load_model(
    "Potato_Models/potato_model.keras"
)

CLASSES = ["Early Blight", "Late Blight", "Healthy"]


def read_file_as_image(data: bytes) -> np.ndarray:
    try:
        image = np.array(Image.open(BytesIO(data)))
        return image
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASSES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    payload = {
        "class": predicted_class,
        "confidence": confidence
    }

    if ENDPOINT_URL:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(ENDPOINT_URL, json=payload)

    return payload
