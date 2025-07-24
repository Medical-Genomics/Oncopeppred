import numpy as np
import tensorflow as tf
from feature_extractor import extract_features
import joblib  # Make sure this is installed

# Load the scaler with error handling
def load_scaler():
    try:
        scaler = joblib.load("scaler11.pkl")  # Make sure the path is correct
        print("✅ Scaler loaded successfully.")
        return scaler
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        return None

# Load the trained CNN model with error handling
def load_model():
    try:
        model = tf.keras.models.load_model("acp_model11.keras", compile=False)
        print(f"✅ Model loaded successfully. Expected input shape: {model.input_shape}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Initialize the scaler and model
scaler = load_scaler()
model = load_model()

# Function to predict whether a peptide sequence is ACP or not
def predict_acp(sequence):
    if model is None or scaler is None:
        return "Error: Model or Scaler not loaded."

    features = extract_features(sequence)
    if features is None or len(features) == 0:
        return "Error: Feature extraction failed."
    
    # Ensure that the feature array is in the correct shape for the model
    features = np.array(features).reshape(1, -1)
    
    try:
        # Transform features using the scaler and reshape to match the model input
        features = scaler.transform(features).reshape(1, 456, 1)
    except Exception as e:
        return f"Error in scaling features: {e}"

    try:
        # Make the prediction using the model
        prediction = model.predict(features)
        print("Model prediction score:", prediction[0][0])
        predicted_class = "ACP" if np.argmax(prediction) == 0 else "Non-ACP"
        return predicted_class
    except Exception as e:
        return f"Error during prediction: {e}"

