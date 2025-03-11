import os
import pickle
import numpy as np
from flask import Flask, render_template, request

# Define function to safely load pickle files
def load_pickle(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"Error: '{file_path}' not found! Please check the file path.")

# Corrected model path (Use raw string to avoid backslash issues in Windows)
# Corrected model path
MODEL_PATH = r"C:\Users\abc\Desktop\python\Ml projects\AAproject\model.pkl"
# OR
# MODEL_PATH = "C:/Users/abc/Desktop/python/Ml projects/AAproject/model.pkl"


# Load model
try:
    model = load_pickle(MODEL_PATH)
except FileNotFoundError as e:
    print(e)
    model = None

# Create Flask app
app = Flask(__name__)

# Crop dictionary for predictions
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
    7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
    12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
    17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
    21: "Chickpea", 22: "Coffee"
}

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        if model is None:
            return render_template('index.html', result="Error: Model is not loaded.")

        try:
            # Extract input features from form
            features = [float(x) for x in request.form.values()]
            single_pred = np.array(features).reshape(1, -1)

            # Model prediction
            prediction = model.predict(single_pred)

            # Get predicted crop
            crop = crop_dict.get(int(prediction[0]), "Unknown crop")
            result = f"{crop} is the best crop to be cultivated right there."
        
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
