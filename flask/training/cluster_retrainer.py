import os, re, cv2, joblib
import numpy as np
from sklearn.decomposition import PCA
import hdbscan
from tensorflow.keras.models import load_model

IMG_SIZE = (224, 224)
MAX_HOUR = 360
SEED = 42
MIN_CLUSTER_SIZE = 50

def retrain_cluster_model(data_folder, encoder_path, output_version_dir):
    print("📦 Loading encoder...")
    encoder = load_model(encoder_path)

    # === Load + process images ===
    print("🔍 Extracting features...")
    features, hours, file_paths = [], [], []
    pattern = re.compile(r'test(\d+)_h(\d+)_\d+')

    for fname in os.listdir(data_folder):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        match = pattern.match(fname)
        if not match:
            continue
        hour = int(match.group(2))
        if hour > MAX_HOUR:
            continue

        img_path = os.path.join(data_folder, fname)
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        latent = encoder.predict(img, verbose=0).flatten()
        norm_hour = hour / MAX_HOUR
        extended = np.append(latent, norm_hour * 2500)

        features.append(extended)
        hours.append(hour)
        file_paths.append(img_path)

    features = np.array(features)
    print(f"✅ Extracted {features.shape}")

    # === PCA ===
    pca = PCA(n_components=100, random_state=SEED)
    reduced = pca.fit_transform(features)
    joblib.dump(pca, os.path.join(output_version_dir, "pca_model.pkl"))

    # === Clustering ===
    print("🧠 Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, prediction_data=True)
    labels = clusterer.fit_predict(reduced)
    joblib.dump(clusterer, os.path.join(output_version_dir, "hdbscan_clusterer.pkl"))

    # === Save clustered images ===
    cluster_dir = os.path.join(output_version_dir, "clustered_images")
    os.makedirs(cluster_dir, exist_ok=True)
    for path, label in zip(file_paths, labels):
        name = os.path.basename(path)
        label_folder = "noise" if label == -1 else f"cluster_{label}"
        dst = os.path.join(cluster_dir, label_folder)
        os.makedirs(dst, exist_ok=True)
        cv2.imwrite(os.path.join(dst, name), cv2.imread(path))

    print(f"✅ Clustering complete. Saved in: {cluster_dir}")
    return labels
