import os
import tempfile
import shutil
from datetime import datetime
from training.retrain import full_retrain_pipeline

# Global dictionary to track training jobs
training_jobs = {}

def background_training(job_id):
    """Background training function - simplified to only use database data"""
    job_temp_dir = None
    try:
        print(f"🚀 Starting background training job {job_id}")
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["message"] = "Training in progress..."
        
        # Create unique temp directory for this job
        job_temp_dir = tempfile.mkdtemp(prefix=f"training_job_{job_id[:8]}_")
        training_jobs[job_id]["temp_dir"] = job_temp_dir
        print(f"📁 Created temp directory for job {job_id}: {job_temp_dir}")
        
        version_dir = full_retrain_pipeline(job_temp_dir=job_temp_dir)
        
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["version"] = os.path.basename(version_dir)
        training_jobs[job_id]["version_path"] = version_dir
        training_jobs[job_id]["message"] = f"Training completed successfully! Version: {os.path.basename(version_dir)}"
        print(f"✅ Background training job {job_id} completed")
        
    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)
        training_jobs[job_id]["message"] = f"Training failed: {str(e)}"
        print(f"❌ Background training job {job_id} failed: {str(e)}")
    
    finally:
        # Clean up job-specific temp directory
        if job_temp_dir and os.path.exists(job_temp_dir):
            try:
                shutil.rmtree(job_temp_dir)
                print(f"🧹 Cleaned up temp directory for job {job_id}")
                if job_id in training_jobs:
                    training_jobs[job_id].pop("temp_dir", None)
            except Exception as e:
                print(f"⚠️ Could not clean up temp directory for job {job_id}: {e}")