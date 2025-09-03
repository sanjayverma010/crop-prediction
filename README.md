# Crop-Recommendation-System

A Machine Learning based web application that recommends the most suitable crop for cultivation based on environmental and soil parameters.  

Built with **Flask, Scikit-learn, Pandas, Numpy**, and deployed using **Render**.  

---

## Features
- Predicts the best crop based on user input (N, P, K values, temperature, humidity, rainfall, pH).
- Interactive web interface.
- Pre-trained ML model for accurate recommendations.
- Lightweight and easy to deploy.

---

## Tech Stack
- **Frontend:** HTML, CSS, Bootstrap (via Flask templates)
- **Backend:** Flask (Python)
- **ML Model:** Scikit-learn
- **Deployment:** Render / Vercel

---

## Project Structure

crop-prediction/
│── app.py # Main Flask application
│── model.pkl # Trained ML model
│── requirements.txt # Dependencies
│── templates/ # HTML templates (home.html, result.html, etc.)
│── static/ # CSS, JS, images
│── README.md # Project documentation


---

## ⚡ Installation & Setup

## Clone the repository**
   ```bash
   git clone https://github.com/sanjayverma010/crop-prediction.git
   cd crop-prediction


## Create virtual environment (recommended)

python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows


## Install dependencies

pip install -r requirements.txt


## Run the application

python app.py


## Open in browser

http://127.0.0.1:5000/


## Deployment (Render)

Push your code to GitHub (make sure requirements.txt and model.pkl are included).

Create a new Web Service in Render
.

## Build Command:

pip install -r requirements.txt


## Start Command:

gunicorn app:app

 

## License

This project is open-source and available under the MIT License.

## Author

Sanjay Verma
Email: your-email@example.com

GitHub: sanjayverma010