import tensorflow as tf
from tensorflow.keras.models import load_model
import os

print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir("."))

try:
    model = load_model('lstm_model.keras')
    print("Model loaded successfully!")
    print("Model summary:")
    model.summary()
except Exception as e:
    print("Error loading .keras model:", e)

try:
    model = load_model('lstm_model.h5')
    print("H5 Model loaded successfully!")
    print("Model summary:")
    model.summary()
except Exception as e:
    print("Error loading .h5 model:", e)