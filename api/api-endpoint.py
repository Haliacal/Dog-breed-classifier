from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uvicorn 
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()

labels = pd.read_csv('classes.csv', index_col='index')
model = tf.keras.models.load_model('models/2')

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.get("/ping")
async def ping():
    return "Hello world!"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = read_file_as_image(await file.read())
    img = tf.image.resize(img, [200, 200]) 
    img = tf.cast(img, tf.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img, verbose=0)[0]
    index = np.argmax(predictions)
    breed = labels.iloc[index,0]
    confidence = predictions[index]*100

    return {
        'class': breed, 
        'confidence': confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)