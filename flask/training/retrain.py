import uuid
import os
import shutil
from database.data import fetch_pocketbase_data
from training.splitter import split_by_hour
from training.train_model import train_hybrid_model
from training.cluster_retrainer import retrain_cluster_model
from datetime import datetime

def full_retrain_pipeline(num_classes: int = 14, job_temp_dir: str = None, hybrid_epochs: int = 1, autoencoder_epochs: int = 1):
    """
    Execute the complete machine learning model retraining pipeline.
    
    This function orchestrates the entire process of updating both classification
    and clustering models using the latest training data from the database.
    The pipeline includes data fetching, preprocessing, model training, and
    version management for reproducible deployments.
    
    Args:
        num_classes (int): Number of growth stage classes to predict (default: 14 days)
        job_temp_dir (str): Isolated temporary directory for this training job
        hybrid_epochs (int): Training epochs for the classification model (default: 1)
        autoencoder_epochs (int): Training epochs for the feature encoder (default: 1)
        
    Returns:
        str: Path to the directory containing all trained models for this version
        
    Raises:
        ValueError: If no training data is found in the database
    """
    # === Workspace Setup ===
    # Use current directory as fallback if no specific temp directory provided
    if job_temp_dir is None:
        job_temp_dir = "."
    
    # Create subdirectories within the job temp directory for organized processing
    training_data_dir = os.path.join(job_temp_dir, "fetched_training_data")    # Raw images from database
    labeled_data_dir = os.path.join(job_temp_dir, "labeled_training_data")     # Images organized by class
    
    # === Step 1: Data Acquisition ===
    # Fetch only images marked as training data from the PocketBase database
    print("📡 Fetching training data from PocketBase...")
    fetch_pocketbase_data(training_data_only=True, output_dir=training_data_dir)
    
    # Validate that training data was successfully retrieved
    if not os.path.exists(training_data_dir) or not os.listdir(training_data_dir):
        raise ValueError("❌ No training data found in database!")
    
    print(f"✅ Found {len(os.listdir(training_data_dir))} training images")

    # === Step 2: Data Preprocessing ===
    # Organize images into class folders based on time progression for supervised learning
    print("📊 Splitting images by hour for classification...")
    split_by_hour(training_data_dir, labeled_data_dir, hour_split=24)

    # === Step 3: Classification Model Training ===
    # Train the hybrid neural network for growth stage prediction
    print(f"🧠 Training classification model... (AE: {autoencoder_epochs}, Hybrid: {hybrid_epochs})")
    
    # Generate unique version identifier for this training run
    version = f"v{uuid.uuid4().hex[:6]}"
    
    # Train the hybrid model with specified epochs and save to version directory
    version_dir = train_hybrid_model(
        labeled_data_dir,                    # Input: Images organized by class
        version=version,                     # Unique version identifier
        num_classes=num_classes,            # Number of growth stages to predict
        hybrid_epochs=hybrid_epochs,        # Classification model training duration
        autoencoder_epochs=autoencoder_epochs # Feature encoder training duration
    )

    # === Step 4: Clustering Model Training ===
    # Train the unsupervised clustering model using the newly trained encoder
    print("🔗 Training clustering model...")
    
    # Use the encoder from the classification model for feature extraction
    encoder_path = os.path.join(version_dir, "encoder_model.keras")
    
    # Train clustering model and save to the same version directory
    retrain_cluster_model(training_data_dir, encoder_path, version_dir)

    # === Pipeline Completion ===
    print(f"✅ Full retraining pipeline completed! Version: {version}")
    print(f"📂 All models saved to: {version_dir}")
    
    # Return path to version directory containing all trained models
    return version_dir