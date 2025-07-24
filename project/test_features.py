import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from feature_extractor import extract_features

# Load trained model
model = load_model("trained_model.keras", compile=False)

# Sample peptide sequence (known ACP)
seq = "KGIRDWLRGFNKLRKWFLRLIKGL"

# Extract features
features = extract_features(seq)  # Should return 456 features

# Reshape features for CNN input (batch_size, 456, 1)
features = np.array(features).reshape(1, 456, 1)

# Debug: print a few feature values
print("🔍 Feature values (first 10):", features.flatten()[:10])

# Predict
prediction = model.predict(features)

# Print predicted probability (assuming binary classification)
print("✅ Predicted probability:", prediction[0][0])
