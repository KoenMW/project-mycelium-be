import numpy as np
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import hdbscan
import joblib
from sklearn.decomposition import PCA
from hdbscan.prediction import approximate_predict

# === Config ===
ENCODER_PATH = "../models/encoder_model.keras"
CLUSTERER_PATH = "../models/hdbscan_clusterer.pkl"
IMG_SIZE = (224, 224)
MAX_HOUR = 360
PCA_COMPONENTS = 100  
SEED = 42

# === Load models ===
print("📦 Loading encoder and HDBSCAN clusterer...")
encoder = load_model(ENCODER_PATH)
clusterer = joblib.load(CLUSTERER_PATH)

pca = None

def load_pca(pca_path="../models/pca_model.pkl"):
    global pca
    if os.path.exists(pca_path):
        print("📉 Loading PCA model...")
        pca = joblib.load(pca_path)
    else:
        raise FileNotFoundError("PCA model not found. Persist it during training for production.")

load_pca()

# === Clustering function ===
def cluster_image(image_bytes, hour=None):
    try:
        image = Image.open(image_bytes).convert("RGB")
        image = image.resize(IMG_SIZE)
        array = img_to_array(image) / 255.0
        array = np.expand_dims(array, axis=0)

        latent = encoder.predict(array, verbose=0).flatten()
        norm_hour = (hour / MAX_HOUR) if hour else 0.0
        feature = np.append(latent, norm_hour * 2500)

        feature = pca.transform([feature])  # PCA to 100D
        cluster_label, probas = approximate_predict(clusterer, feature)
        cluster_label = int(cluster_label)
        confidence = float(np.max(probas))
        return {
            "cluster": cluster_label,
            "confidence": round(confidence, 4),
            "hour": hour,
            "normalized_hour": round(norm_hour, 4)
        }, None
    except Exception as e:
        return None, str(e)
