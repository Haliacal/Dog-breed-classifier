from google.cloud import storage
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO


BUCKET_NAME = "dog-breed-classifier"

interpreter = None
labels = None
model = None

def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def predict_using_tfite(image):
    image = tf.image.resize(image, [200, 200]) 
    img = tf.cast(image, tf.float32) / 255.0
    test_image = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()
    output = interpreter.tensor(output_index)
    predictions = output()[0]

    index = np.argmax(predictions)
    breed = labels.iloc[index,0]
    confidence = round(predictions[index]*100, 2)
    return breed, confidence


def predict(request):
    global interpreter
    global input_index
    global output_index
    global labels

    if(not interpreter):
        download_blob(
            BUCKET_NAME,
            "Models/model.tflite",
            "/tmp/model.tflite",
        )

        download_blob(
            BUCKET_NAME,
            "Labels/labels.csv",
            "/tmp/labels.csv"
        )
        labels = pd.read_csv('/tmp/labels.csv', index_col='index')

    interpreter = tf.lite.Interpreter(model_path="/tmp/model.tflite")
    
    
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    image = request.files["file"]

    image = np.array(Image.open(image).convert("RGB"))[:, :, ::-1]
    breed, confidence = predict_using_tfite(image)

    return {"class": breed, "confidence": confidence}