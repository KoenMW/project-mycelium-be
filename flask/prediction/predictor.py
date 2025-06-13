import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import io
import os

# === Constants ===
DEFAULT_MODEL_PATH = "models/best_hybrid_model.keras"
IMG_SIZE = (224, 224)
CLASSES = [str(i) for i in range(14)]  # 0–13

# === Model cache ===
_models = {}

def load_prediction_model(version="default"):
    """Load prediction model for specific version"""
    if version in _models:
        return _models[version]
    
    if version == "default":
        model_path = DEFAULT_MODEL_PATH
    else:
        model_path = os.path.join("model_versions", version, "hybrid_model.keras")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found for version '{version}' at {model_path}")
    
    print(f"🔍 Loading prediction model for version: {version}")
    model = load_model(model_path)
    _models[version] = model
    print(f"✅ Model loaded for version: {version}")
    return model

# === Load default model at startup ===
try:
    load_prediction_model("default")
except FileNotFoundError:
    print("⚠️ Default prediction model not found")

# === Prediction function ===
def predict_growth_stage(image_bytes, version="default"):
    try:
        model = load_prediction_model(version)
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMG_SIZE)
        array = img_to_array(image) / 255.0  # Normalize to [0,1]
        input_data = np.expand_dims(array, axis=0)

        # Feed into both VGG and encoder inputs
        predictions = model.predict([input_data, input_data])
        predicted_index = int(np.argmax(predictions, axis=1)[0])
        predicted_class = CLASSES[predicted_index]
        confidence = float(np.max(predictions))

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "version": version
        }, None
    except Exception as e:
        return None, str(e)