import os
import tempfile
import shutil
from datetime import datetime
from training.retrain import full_retrain_pipeline

# === Global Job Tracking ===
# Dictionary to store status and metadata for all active/completed training jobs
# Key: job_id (string), Value: dict with status, messages, paths, etc.
training_jobs = {}

def background_training(job_id, hybrid_epochs=1, autoencoder_epochs=1):
    """
    Execute machine learning model training in the background as a separate job.
    
    This function manages the complete training pipeline execution, including:
    - Job status tracking and progress reporting
    - Temporary directory management for training artifacts
    - Error handling and cleanup
    - Integration with the full retraining pipeline
    
    Args:
        job_id (str): Unique identifier for this training job (used for tracking)
        hybrid_epochs (int): Number of training epochs for the hybrid classification model
        autoencoder_epochs (int): Number of training epochs for the autoencoder feature extractor
        
    Returns:
        None: Updates global training_jobs dictionary with progress and results
    """
    job_temp_dir = None  # Initialize temp directory variable for cleanup
    
    try:
        # === Job Initialization ===
        print(f"🚀 Starting background training job {job_id}")
        print(f"📊 Epochs: Autoencoder={autoencoder_epochs}, Hybrid={hybrid_epochs}")
        
        # Update job status to indicate training has begun
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["message"] = f"Training in progress... (AE: {autoencoder_epochs}, Hybrid: {hybrid_epochs})"
        
        # === Temporary Directory Setup ===
        # Create isolated workspace for this training job to avoid conflicts
        job_temp_dir = tempfile.mkdtemp(prefix=f"training_job_{job_id[:8]}_")
        training_jobs[job_id]["temp_dir"] = job_temp_dir
        print(f"📁 Created temp directory for job {job_id}: {job_temp_dir}")
        
        # === Execute Training Pipeline ===
        # Run the complete model retraining process with specified parameters
        # This includes data preparation, model training, evaluation, and saving
        version_dir = full_retrain_pipeline(
            job_temp_dir=job_temp_dir,           # Isolated workspace for this job
            hybrid_epochs=hybrid_epochs,         # Classification model training duration
            autoencoder_epochs=autoencoder_epochs # Feature extractor training duration
        )
        
        # === Success Handling ===
        # Update job status with successful completion details
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["version"] = os.path.basename(version_dir)      # New model version name
        training_jobs[job_id]["version_path"] = version_dir                   # Full path to trained models
        training_jobs[job_id]["message"] = f"Training completed successfully! Version: {os.path.basename(version_dir)} (AE: {autoencoder_epochs}, Hybrid: {hybrid_epochs})"
        print(f"✅ Background training job {job_id} completed")
        
    except Exception as e:
        # === Error Handling ===
        # Capture and store any training failures for user feedback
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)                              # Detailed error message
        training_jobs[job_id]["message"] = f"Training failed: {str(e)}"      # User-friendly error summary
        print(f"❌ Background training job {job_id} failed: {str(e)}")
    
    finally:
        # === Cleanup Process ===
        # Always attempt to clean up temporary files, regardless of success/failure
        if job_temp_dir and os.path.exists(job_temp_dir):
            try:
                # Remove all temporary training files and directories
                shutil.rmtree(job_temp_dir)
                print(f"🧹 Cleaned up temp directory for job {job_id}")
                
                # Remove temp directory path from job tracking (no longer needed)
                if job_id in training_jobs:
                    training_jobs[job_id].pop("temp_dir", None)
                    
            except Exception as e:
                # Log cleanup failures but don't crash the job
                print(f"⚠️ Could not clean up temp directory for job {job_id}: {e}")