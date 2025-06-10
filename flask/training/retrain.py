import uuid
import os
import shutil
from database.data import fetch_pocketbase_data
from training.splitter import split_by_hour
from training.train_model import train_hybrid_model
from training.cluster_retrainer import retrain_cluster_model
from datetime import datetime

def full_retrain_pipeline(num_classes: int = 14, job_temp_dir: str = None):
    """
    Retrain models using only database data where trainingData=true
    Args:
        num_classes: Number of classification classes
        job_temp_dir: Temporary directory for this specific job
    """
    if job_temp_dir is None:
        job_temp_dir = "."
    
    # Create subdirectories within the job temp directory
    training_data_dir = os.path.join(job_temp_dir, "fetched_training_data")
    labeled_data_dir = os.path.join(job_temp_dir, "labeled_training_data")
    
    # 1. Fetch only training data from PocketBase
    print("📡 Fetching training data from PocketBase...")
    fetch_pocketbase_data(training_data_only=True, output_dir=training_data_dir)
    
    # Check if we have training data
    if not os.path.exists(training_data_dir) or not os.listdir(training_data_dir):
        raise ValueError("❌ No training data found in database!")
    
    print(f"✅ Found {len(os.listdir(training_data_dir))} training images")

    # 2. Split images by hour for classification
    print("📊 Splitting images by hour for classification...")
    split_by_hour(training_data_dir, labeled_data_dir, hour_split=24)

    # 3. Train prediction model
    print("🧠 Training classification model...")
    version = f"v{uuid.uuid4().hex[:6]}"
    version_dir = train_hybrid_model(labeled_data_dir, version=version, num_classes=num_classes)

    # 4. Train clustering model
    print("🔗 Training clustering model...")
    encoder_path = os.path.join(version_dir, "encoder_model.keras")
    retrain_cluster_model(training_data_dir, encoder_path, version_dir)

    print(f"✅ Full retraining pipeline completed! Version: {version}")
    return version_dir