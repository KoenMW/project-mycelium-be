import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image
import io
# ...existing code...

# === Constants ===
MODEL_PATH = "../models/best_hybrid_model.keras"
IMG_SIZE = (224, 224)
CLASSES = [str(i) for i in range(14)]  # 0–13

# === Load model once at startup ===
print("🔍 Loading prediction model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded.")

# === Prediction function ===
def predict_growth_stage(image_bytes):
    try:
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
            "confidence": round(confidence, 4)
        }, None
    except Exception as e:
        return None, str(e)
