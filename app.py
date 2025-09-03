import os
import pickle
import joblib
import requests
from io import BytesIO
import numpy as np
from flask import Flask, render_template, request
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# ==============================
# Load Model from URL (instead of local system)
# ==============================
MODEL_URL = "https://raw.githubusercontent.com/sanjayverma010/crop-prediction/main/model.pkl"

def load_model_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return joblib.load(BytesIO(response.content))
    except Exception as e:
        print(f"⚠️ Failed to load model from URL: {e}")
        return None

# Load model
model = load_model_from_url(MODEL_URL)

# ==============================
# Flask App
# ==============================
app = Flask(__name__)

# Crop Dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
    7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
    12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
    17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
    21: "Chickpea", 22: "Coffee"
}

# ==============================
# Routes
# ==============================
@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        if model is None:
            return render_template('index.html', result="⚠️ Model not loaded. Check MODEL_URL.")

        try:
            # Extract features
            features = [float(x) for x in request.form.values()]
            
            if len(features) != 7:
                raise ValueError("⚠️ Incorrect number of features. Expected 7.")

            single_pred = np.array(features).reshape(1, -1)

            # Prediction
            prediction = model.predict(single_pred)

            # Map crop name
            crop = crop_dict.get(int(prediction[0]), "Unknown crop")
            result = f"🌱 {crop} is the best crop to be cultivated right there."
        
        except Exception as e:
            result = f"⚠️ Error: {str(e)}"

    return render_template('index.html', result=result)


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
