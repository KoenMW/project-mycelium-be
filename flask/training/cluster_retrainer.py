import os, re, cv2, joblib
import numpy as np
from sklearn.decomposition import PCA
import hdbscan
from tensorflow.keras.models import load_model

# === Configuration Constants ===
IMG_SIZE = (224, 224)    # Standard image size for feature extraction
MAX_HOUR = 360           # Maximum hour value for filtering (15 days * 24 hours)
SEED = 42               # Random seed for reproducible results
MIN_CLUSTER_SIZE = 50   # Minimum number of images required to form a cluster

def retrain_cluster_model(data_folder, encoder_path, output_version_dir):
    """
    Retrain the clustering model using new training data.
    
    This function performs the complete clustering pipeline:
    1. Load pre-trained encoder for feature extraction
    2. Process all training images and extract features
    3. Apply PCA for dimensionality reduction
    4. Train HDBSCAN clustering model
    5. Organize clustered images into folders
    
    Args:
        data_folder (str): Directory containing training images with standardized naming
        encoder_path (str): Path to the trained encoder model for feature extraction
        output_version_dir (str): Directory to save the new clustering models and results
        
    Returns:
        np.array: Cluster labels assigned to each image (-1 for noise)
    """
    # === Load Pre-trained Encoder ===
    # The encoder extracts high-level features from images for clustering
    print("📦 Loading encoder...")
    encoder = load_model(encoder_path)

    # === Image Processing and Feature Extraction ===
    print("🔍 Extracting features...")
    features, hours, file_paths = [], [], []
    
    # Regex pattern to extract metadata from standardized filenames
    # Expected format: test{num}_h{hour}_{angle}.jpg
    pattern = re.compile(r'test(\d+)_h(\d+)_\d+')

    # Process each image file in the training data folder
    for fname in os.listdir(data_folder):
        # === File Filtering ===
        # Only process image files
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
            
        # Extract hour information from filename using regex
        match = pattern.match(fname)
        if not match:
            continue  # Skip files that don't match naming convention
            
        hour = int(match.group(2))
        
        # Filter out images beyond the maximum time range
        if hour > MAX_HOUR:
            continue

        # === Image Preprocessing ===
        # Load and prepare image for feature extraction
        img_path = os.path.join(data_folder, fname)
        img = cv2.imread(img_path)           # Load image in BGR format
        img = cv2.resize(img, IMG_SIZE)      # Resize to standard dimensions
        img = img.astype(np.float32) / 255.0 # Normalize pixel values to [0,1]
        img = np.expand_dims(img, axis=0)    # Add batch dimension for encoder

        # === Feature Extraction ===
        # Use encoder to extract high-level features from the image
        latent = encoder.predict(img, verbose=0).flatten()
        
        # === Temporal Feature Engineering ===
        # Add normalized hour as an additional feature
        norm_hour = hour / MAX_HOUR  # Normalize to [0,1] range
        # Append temporal feature with scaling factor (2500 from training)
        extended = np.append(latent, norm_hour * 2500)

        # Store extracted features and metadata
        features.append(extended)
        hours.append(hour)
        file_paths.append(img_path)

    # Convert feature list to numpy array for efficient processing
    features = np.array(features)
    print(f"✅ Extracted {features.shape}")

    # === Dimensionality Reduction with PCA ===
    # Reduce feature dimensions to improve clustering performance and speed
    print("🔄 Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=100, random_state=SEED)  # Reduce to 100 components
    reduced = pca.fit_transform(features)           # Fit PCA and transform features
    
    # Save PCA model for future use with new data
    pca_path = os.path.join(output_version_dir, "pca_model.pkl")
    joblib.dump(pca, pca_path)
    print(f"💾 Saved PCA model: {pca_path}")

    # === HDBSCAN Clustering ===
    # Train hierarchical density-based clustering model
    print("🧠 Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,  # Minimum images needed to form a cluster
        prediction_data=True                # Enable predictions on new data
    )
    
    # Fit clustering model and get cluster labels
    labels = clusterer.fit_predict(reduced)
    
    # Save clustering model for future predictions
    clusterer_path = os.path.join(output_version_dir, "hdbscan_clusterer.pkl")
    joblib.dump(clusterer, clusterer_path)
    print(f"💾 Saved HDBSCAN model: {clusterer_path}")

    # === Organize Clustered Images ===
    # Create directory structure to visualize clustering results
    print("📁 Organizing clustered images...")
    cluster_dir = os.path.join(output_version_dir, "clustered_images")
    os.makedirs(cluster_dir, exist_ok=True)
    
    # Copy images to cluster-specific folders for visual inspection
    for path, label in zip(file_paths, labels):
        name = os.path.basename(path)
        
        # Handle noise points (label -1) and regular clusters
        label_folder = "noise" if label == -1 else f"cluster_{label}"
        
        # Create cluster folder if it doesn't exist
        dst = os.path.join(cluster_dir, label_folder)
        os.makedirs(dst, exist_ok=True)
        
        # Copy image to appropriate cluster folder
        dst_path = os.path.join(dst, name)
        cv2.imwrite(dst_path, cv2.imread(path))

    # === Results Summary ===
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude noise
    n_noise = np.sum(labels == -1)
    
    print(f"✅ Clustering complete!")
    print(f"📊 Found {n_clusters} clusters with {n_noise} noise points")
    print(f"📂 Results saved in: {cluster_dir}")
    
    return labels