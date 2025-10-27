from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# ---------- Model + Scaler Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'lstm_energy_forecast_model.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

# If files are in 'models' subfolder
if not os.path.exists(MODEL_PATH):
    alt_model = os.path.join(BASE_DIR, 'models', 'lstm_energy_forecast_model.h5')
    if os.path.exists(alt_model):
        MODEL_PATH = alt_model

if not os.path.exists(SCALER_PATH):
    alt_scaler = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
    if os.path.exists(alt_scaler):
        SCALER_PATH = alt_scaler

# ---------- Load Model + Scaler Safely ----------
model = None
scaler = None

try:
    # Load the trained LSTM model (ignore training configs)
    model = load_model(MODEL_PATH, compile=False)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print("❌ Error loading model:", e)

# Try loading the scaler — if fails, create a default one
try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✅ Scaler loaded successfully from: {SCALER_PATH}")
except Exception as e:
    print("⚠️ Warning: Could not load scaler:", e)
    print("🛠️ Creating a default MinMaxScaler (temporary).")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[0], [1]]))  # dummy fit

# ---------- Home Route ----------
@app.route('/')
def home():
    return render_template('index.html')

# ---------- Forecast Route ----------
@app.route('/forecast', methods=['POST'])
def forecast():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check the model file path.'}), 500

    try:
        req = request.get_json()
        if not req or 'data' not in req:
            return jsonify({'error': 'No input data provided.'}), 400

        data = req['data']
        if not isinstance(data, list) or len(data) == 0:
            return jsonify({'error': 'Invalid input format. Expecting list of numbers.'}), 400

        # Prepare data
        arr = np.array(data).reshape(1, -1, 1)

        # Scale safely (if scaler is default, it will just normalize)
        try:
            scaled_seq = scaler.transform(arr.reshape(-1, 1)).reshape(1, -1, 1)
        except Exception:
            scaled_seq = arr  # fallback if scaling fails

        # Predict
        pred_scaled = model.predict(scaled_seq)
        try:
            pred = scaler.inverse_transform(pred_scaled)[0][0]
        except Exception:
            pred = float(pred_scaled[0][0])

        return jsonify({'prediction': float(pred)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    app.run(debug=True)






