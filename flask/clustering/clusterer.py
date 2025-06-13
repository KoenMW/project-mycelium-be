import numpy as np
import os

# === Configuration Constants ===
# Default file paths for the pre-trained machine learning models
DEFAULT_ENCODER_PATH = "../models/encoder_model.keras"  # Neural network encoder for feature extraction
DEFAULT_CLUSTERER_PATH = "../models/hdbscan_clusterer.pkl"  # HDBSCAN clustering model
DEFAULT_PCA_PATH = "../models/pca_model.pkl"  # Principal Component Analysis model for dimensionality reduction

# Image processing and feature engineering parameters
IMG_SIZE = (224, 224)  # Standard size to resize all input images
MAX_HOUR = 360  # Maximum hour value for normalization (15 days * 24 hours)
PCA_COMPONENTS = 100  # Number of components to reduce features to using PCA
SEED = 42  # Random seed for reproducibility

# === Global Model Cache ===
# These dictionaries store loaded models in memory to avoid reloading them repeatedly
_encoders = {}     # Cache for encoder models (different versions)
_clusterers = {}   # Cache for clustering models (different versions)
_pcas = {}         # Cache for PCA models (different versions)

# === Conditional Import System ===
# Only import heavy ML libraries when not in testing mode to speed up tests
# This allows the application to run tests without loading large model files
if not os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
    from keras.models import load_model  # For loading neural network models
    from keras.preprocessing.image import img_to_array  # Convert PIL images to numpy arrays
    from PIL import Image  # Python Imaging Library for image processing
    import hdbscan  # Hierarchical density-based clustering algorithm
    import joblib  # For loading scikit-learn models (PCA, clustering)
    from sklearn.decomposition import PCA  # Principal Component Analysis
    from hdbscan.prediction import approximate_predict  # For predicting cluster membership
else:
    print("🧪 Skipping ML library imports for testing")

def load_clustering_models(version="default"):
    """
    Load and cache clustering models for a specific version.
    
    This function handles loading three interconnected models:
    1. Encoder: Extracts features from images using a neural network
    2. PCA: Reduces dimensionality of features for efficient clustering
    3. Clusterer: Groups similar images based on their features
    
    Args:
        version (str): Model version to load ("default" or specific version name)
        
    Returns:
        tuple: (encoder_model, clusterer_model, pca_model) or (None, None, None) if testing
    """
    # Skip model loading entirely during testing to improve test speed
    if os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
        print(f"🧪 Skipping clustering model loading for testing - version: {version}")
        return None, None, None
    
    # Check if models for this version are already cached in memory
    if version in _encoders:
        return _encoders[version], _clusterers[version], _pcas[version]
    
    # Determine file paths based on version
    if version == "default":
        # Use default model paths for the main production models
        encoder_path = DEFAULT_ENCODER_PATH
        clusterer_path = DEFAULT_CLUSTERER_PATH
        pca_path = DEFAULT_PCA_PATH
    else:
        # Use version-specific directory structure for experimental or alternative models
        version_dir = os.path.join("model_versions", version)
        encoder_path = os.path.join(version_dir, "encoder_model.keras")
        clusterer_path = os.path.join(version_dir, "hdbscan_clusterer.pkl")
        pca_path = os.path.join(version_dir, "pca_model.pkl")
    
    # Validate that all required model files exist before attempting to load
    missing_files = []
    for name, path in [("encoder", encoder_path), ("clusterer", clusterer_path), ("pca", pca_path)]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    # Raise an error if any model files are missing
    if missing_files:
        raise FileNotFoundError(f"Missing model files for version '{version}': {', '.join(missing_files)}")
    
    print(f"📦 Loading clustering models for version: {version}")
    
    # Load the three models from their respective files
    encoder = load_model(encoder_path)      # Keras neural network model
    clusterer = joblib.load(clusterer_path)  # HDBSCAN clustering model
    pca = joblib.load(pca_path)             # PCA dimensionality reduction model
    
    # Cache the loaded models in memory for future use
    _encoders[version] = encoder
    _clusterers[version] = clusterer
    _pcas[version] = pca
    
    print(f"✅ Clustering models loaded for version: {version}")
    return encoder, clusterer, pca

# === Startup Model Loading ===
# Automatically load default models when the module is imported (unless testing)
# This ensures models are ready for immediate use without loading delays
if not os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
    try:
        load_clustering_models("default")
    except FileNotFoundError as e:
        print(f"⚠️ Default clustering models not found: {e}")

# === Main Clustering Function ===
def cluster_image(image_bytes, hour=None, version="default"):
    """
    Classify an image into a cluster and return clustering information.
    
    This function processes an image through the complete clustering pipeline:
    1. Load and preprocess the image
    2. Extract features using the encoder neural network
    3. Add temporal information (hour) as an additional feature
    4. Reduce dimensionality using PCA
    5. Predict cluster membership using HDBSCAN
    
    Args:
        image_bytes: Raw image data as bytes (e.g., from file upload)
        hour (int, optional): Hour information (0-360) to include as temporal feature
        version (str): Model version to use for clustering
        
    Returns:
        tuple: (result_dict, error_message)
            - result_dict: Contains cluster, confidence, hour info, and version
            - error_message: String describing any error that occurred, or None
    """
    # Return mock data during testing to avoid model dependencies
    if os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
        return {
            "cluster": 0,
            "confidence": 0.85,
            "hour": hour,
            "normalized_hour": (hour / MAX_HOUR) if hour else 0.0,
            "version": version
        }, None
    
    try:
        # Load the required models for the specified version
        encoder, clusterer, pca = load_clustering_models(version)
        
        # === Image Preprocessing ===
        # Convert bytes to PIL Image and ensure RGB format (remove alpha channel if present)
        image = Image.open(image_bytes).convert("RGB")
        # Resize to standard dimensions expected by the encoder model
        image = image.resize(IMG_SIZE)
        # Convert to numpy array and normalize pixel values to [0,1] range
        array = img_to_array(image) / 255.0
        # Add batch dimension (model expects 4D input: batch_size, height, width, channels)
        array = np.expand_dims(array, axis=0)

        # === Feature Extraction ===
        # Use encoder neural network to extract high-level features from the image
        latent = encoder.predict(array, verbose=0).flatten()
        
        # === Temporal Feature Engineering ===
        # Normalize hour to [0,1] range and scale to match feature magnitude
        norm_hour = (hour / MAX_HOUR) if hour else 0.0
        # Append temporal feature to image features (2500 scaling factor from training)
        feature = np.append(latent, norm_hour * 2500)

        # === Dimensionality Reduction ===
        # Apply PCA to reduce feature dimensions for efficient clustering
        feature = pca.transform([feature])  # Reshape to 2D array for PCA
        
        # === Cluster Prediction ===
        # Predict cluster membership and get confidence probabilities
        cluster_label, probas = approximate_predict(clusterer, feature)
        cluster_label = int(cluster_label)  # Convert to standard Python int
        confidence = float(np.max(probas))  # Highest probability as confidence score
        
        # Return structured result with all relevant information
        return {
            "cluster": cluster_label,
            "confidence": round(confidence, 4),  # Round to 4 decimal places
            "hour": hour,
            "normalized_hour": round(norm_hour, 4),
            "version": version
        }, None
        
    except Exception as e:
        # Return error information if any step in the pipeline fails
        return None, str(e)