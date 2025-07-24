import numpy as np
from keras.models import load_model
import joblib

# Load model and scaler
model = load_model('acp_model11.keras', compile=False)
scaler = joblib.load('scaler11.pkl')  # Assuming you saved your scaler with joblib

# Example: Function to extract features (replace this with your actual feature extraction method)
def extract_features(peptide_sequence):
    # This is just a placeholder. You need to implement your feature extraction logic here.
    # The output should be a 456-length feature vector.
    features = np.zeros(456)  # Example, replace with real feature extraction
    return features

# Example peptide sequence
peptide_sequence = "ATTTGCAAA"

# Extract features from the peptide
input_features = extract_features(peptide_sequence)

# Reshape input to match the model's expected input format
input_features = input_features.reshape(1, -1)

# Scale the input features
scaled_input = scaler.transform(input_features)

# Make prediction
prediction = model.predict(scaled_input)

# Output the prediction
print(f"Prediction: {prediction}")
if prediction[0][0] > prediction[0][1]:
    print("Predicted class: ACP")
else:
    print("Predicted class: Non-ACP")

