from fastapi import FastAPI, File, UploadFile
import uvicorn
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json

app = FastAPI()

# Load model architecture
with open("model.h5", "r") as f:
    model_json = f.read()

model = model_from_json(model_json)

# Load weights
model.load_weights("model_weights.h5")

def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    processed = preprocess(image)
    prediction = model.predict(processed)[0][0]

    result = "Real" if prediction > 0.5 else "AI-Generated"
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


