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
DEFAULT_ENCODER_PATH = "../models/encoder_model.keras"
DEFAULT_CLUSTERER_PATH = "../models/hdbscan_clusterer.pkl"
DEFAULT_PCA_PATH = "../models/pca_model.pkl"
IMG_SIZE = (224, 224)
MAX_HOUR = 360
PCA_COMPONENTS = 100  
SEED = 42

# === Model cache ===
_encoders = {}
_clusterers = {}
_pcas = {}

def load_clustering_models(version="default"):
    """Load clustering models for specific version"""
    if version in _encoders:
        return _encoders[version], _clusterers[version], _pcas[version]
    
    if version == "default":
        encoder_path = DEFAULT_ENCODER_PATH
        clusterer_path = DEFAULT_CLUSTERER_PATH
        pca_path = DEFAULT_PCA_PATH
    else:
        version_dir = os.path.join("model_versions", version)
        encoder_path = os.path.join(version_dir, "encoder_model.keras")
        clusterer_path = os.path.join(version_dir, "hdbscan_clusterer.pkl")
        pca_path = os.path.join(version_dir, "pca_model.pkl")
    
    # Check if all files exist
    missing_files = []
    for name, path in [("encoder", encoder_path), ("clusterer", clusterer_path), ("pca", pca_path)]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        raise FileNotFoundError(f"Missing model files for version '{version}': {', '.join(missing_files)}")
    
    print(f"📦 Loading clustering models for version: {version}")
    
    # Load models
    encoder = load_model(encoder_path)
    clusterer = joblib.load(clusterer_path)
    pca = joblib.load(pca_path)
    
    # Cache models
    _encoders[version] = encoder
    _clusterers[version] = clusterer
    _pcas[version] = pca
    
    print(f"✅ Clustering models loaded for version: {version}")
    return encoder, clusterer, pca

# === Load default models at startup ===
try:
    load_clustering_models("default")
except FileNotFoundError as e:
    print(f"⚠️ Default clustering models not found: {e}")

# === Clustering function ===
def cluster_image(image_bytes, hour=None, version="default"):
    try:
        encoder, clusterer, pca = load_clustering_models(version)
        
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
            "normalized_hour": round(norm_hour, 4),
            "version": version
        }, None
    except Exception as e:
        return None, str(e)