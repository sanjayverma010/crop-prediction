import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Function to safely load pickle files
def load_pickle(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"Error: '{file_path}' not found! Please check the file path.")

# Model Path
MODEL_PATH = "C:/Users/abc/Desktop/python/Ml projects/AAproject/model.pkl"

# Load Model
try:
    model = load_pickle(MODEL_PATH)
    print("✅ Model loaded successfully!")
except FileNotFoundError as e:
    print(e)
    model = None

# Dummy dataset for evaluation (Replace with actual dataset)
try:
    # Ensure the dataset has 7 features as expected by the model
    x = np.random.rand(100, 7)  # Example dataset with 7 features
    y = np.random.randint(1, 23, size=100)  # Crop labels from 1 to 22

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    if model is not None:
        y_pred = model.predict(x_test)

        print("\n🔍 **Model Performance Evaluation:**")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
    else:
        print("⚠️ Error: Model not loaded. Skipping performance evaluation.")

except Exception as e:
    print(f"⚠️ Error during evaluation: {e}")

# Create Flask App
app = Flask(__name__)

# Crop Dictionary for Predictions
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
            return render_template('index.html', result="⚠️ Error: Model is not loaded.")

        try:
            # Extract Input Features
            features = [float(x) for x in request.form.values()]
            
            # Ensure the correct number of features
            if len(features) != 7:
                raise ValueError("⚠️ Incorrect number of features. Expected 7.")

            single_pred = np.array(features).reshape(1, -1)

            # Model Prediction
            prediction = model.predict(single_pred)

            # Get Predicted Crop
            crop = crop_dict.get(int(prediction[0]), "Unknown crop")
            result = f"🌱 {crop} is the best crop to be cultivated right there."
        
        except Exception as e:
            result = f"⚠️ Error: {str(e)}"

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app = app.run(debug=True)