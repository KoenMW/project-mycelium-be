from flask import Flask, request, jsonify, send_file
from random import random as rng
from flask_cors import CORS
from segmentation.segment_mycelium import segment_image, segment_and_save
from prediction.predictor import predict_growth_stage
from clustering.clusterer import cluster_image
from training.retrain import full_retrain_pipeline
import io
import sys
import psutil 
import os
from database.data import fetch_pocketbase_data
import zipfile
import tempfile
import shutil
import threading
import uuid
from datetime import datetime
import glob
app = Flask(__name__)

allowed_origins = [
    "https://koenmw.github.io",
    "http://localhost:5173"
]

CORS(app, origins=allowed_origins)

# Global dictionary to track training jobs
training_jobs = {}

def get_available_versions():
    """Get list of available model versions"""
    versions = []
    model_versions_dir = "model_versions"
    
    if os.path.exists(model_versions_dir):
        for version_folder in os.listdir(model_versions_dir):
            version_path = os.path.join(model_versions_dir, version_folder)
            if os.path.isdir(version_path):
                # Check if required model files exist
                required_files = [
                    "hybrid_model.keras",
                    "encoder_model.keras", 
                    "hdbscan_clusterer.pkl",
                    "pca_model.pkl"
                ]
                
                missing_files = []
                for file in required_files:
                    if not os.path.exists(os.path.join(version_path, file)):
                        missing_files.append(file)
                
                version_info = {
                    "version": version_folder,
                    "path": version_path,
                    "complete": len(missing_files) == 0,
                    "missing_files": missing_files,
                    "created": None
                }
                
                # Try to get creation date
                try:
                    version_info["created"] = datetime.fromtimestamp(
                        os.path.getctime(version_path)
                    ).isoformat()
                except:
                    pass
                
                versions.append(version_info)
    
    # Add default models if they exist
    default_models_dir = "../models"
    if os.path.exists(default_models_dir):
        default_files = [
            "best_hybrid_model.keras",
            "encoder_model.keras",
            "hdbscan_clusterer.pkl", 
            "pca_model.pkl"
        ]
        
        missing_default = []
        for file in default_files:
            if not os.path.exists(os.path.join(default_models_dir, file)):
                missing_default.append(file)
        
        versions.insert(0, {
            "version": "default",
            "path": default_models_dir,
            "complete": len(missing_default) == 0,
            "missing_files": missing_default,
            "created": "N/A"
        })
    
    return versions

def background_training(job_id, upload_folder):
    """Background training function"""
    try:
        print(f"🚀 Starting background training job {job_id}")
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["message"] = "Training in progress..."
        
        version_dir = full_retrain_pipeline(upload_folder)
        
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
        # Cleanup temporary directory if it exists
        temp_dir = training_jobs[job_id].get("temp_dir")
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"🧹 Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"⚠️ Failed to cleanup temp dir {temp_dir}: {e}")

@app.route('/')
def hello():
    return "Server is ok! 👍"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({ "error": "No file uploaded" }), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({ "error": "Empty filename" }), 400

    # Get model version from request
    version = request.form.get('version', 'default')
    
    # Perform segmentation
    segmented_image, error = segment_image(file.read())
    if error:
        return jsonify({ "error": error }), 400
    
    # === Prediction ===
    segmented_image.seek(0)  # Reset stream position
    prediction_result, prediction_error = predict_growth_stage(segmented_image.read(), version=version)
    if prediction_error:
        return jsonify({ "error": prediction_error }), 500

    return jsonify({
        "prediction": prediction_result,
        "model_version": version
    })

@app.route('/cluster', methods=['POST'])
def cluster():
    if 'file' not in request.files:
        return jsonify({ "error": "No file uploaded" }), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({ "error": "Empty filename" }), 400

    # Get model version from request
    version = request.form.get('version', 'default')
    
    # Optional: extract hour (if provided by client)
    hour = request.form.get("hour")
    try:
        hour = int(hour) if hour else None
    except ValueError:
        return jsonify({ "error": "Invalid hour format" }), 400

    result, error = cluster_image(io.BytesIO(file.read()), hour, version=version)
    if error:
        return jsonify({ "error": error }), 500

    return jsonify({ 
        "clustering": result,
        "model_version": version
    })

@app.route('/health')
def health():
    # Check default model files
    default_model_files = [
        "../models/best_hybrid_model.keras",
        "../models/encoder_model.keras",
        "../models/hdbscan_clusterer.pkl",
        "../models/pca_model.pkl",
        "../models/yolo_segmenting_model.pt"
    ]
    missing = [f for f in default_model_files if not os.path.exists(f)]
    status = "ok" if not missing else "degraded"

    # Get available versions
    versions = get_available_versions()
    complete_versions = [v for v in versions if v["complete"]]

    # Optionally, add resource info
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss // (1024 * 1024)  # MB

    return jsonify({
        "status": status,
        "missing_files": missing,
        "python_version": sys.version,
        "memory_usage_mb": mem,
        "model_versions": {
            "total": len(versions),
            "complete": len(complete_versions),
            "versions": versions
        }
    }), 200 if status == "ok" else 503

@app.route('/metadata')
def metadata():
    versions = get_available_versions()
    complete_versions = [v for v in versions if v["complete"]]
    
    return jsonify({
        "project": "mycelium-be",
        "version": "1.0.0",
        "author": "Group 6",
        "description": "API for mycelium project",
        "model_versions": {
            "available": [v["version"] for v in complete_versions],
            "default": "default",
            "total_versions": len(versions),
            "complete_versions": len(complete_versions),
            "details": versions
        },
        "endpoints": {
            "predict": {
                "method": "POST",
                "parameters": ["file (required)", "version (optional, default: 'default')"],
                "description": "Predict growth stage of mycelium image"
            },
            "cluster": {
                "method": "POST", 
                "parameters": ["file (required)", "hour (optional)", "version (optional, default: 'default')"],
                "description": "Cluster mycelium image"
            },
            "retrain": {
                "method": "POST",
                "parameters": ["zip_file (required)"],
                "description": "Retrain models with new data"
            }
        }
    }), 200

# ... rest of your existing routes remain the same ...

@app.route("/retrain", methods=["POST"])
def retrain_with_zip():
    """Start training with uploaded ZIP file"""
    if 'zip_file' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zip_file']
    if zip_file.filename == '' or not zip_file.filename.lower().endswith('.zip'):
        return jsonify({"error": "Please upload a ZIP file"}), 400

    temp_dir = tempfile.mkdtemp(prefix="retrain_zip_")
    
    try:
        # Save uploaded ZIP
        zip_path = os.path.join(temp_dir, "upload.zip")
        zip_file.save(zip_path)
        
        # Extract ZIP
        extract_dir = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find the main folder (should contain subfolders with timestamp names)
        main_folder = None
        for item in os.listdir(extract_dir):
            item_path = os.path.join(extract_dir, item)
            if os.path.isdir(item_path):
                main_folder = item_path
                break
        
        if not main_folder:
            return jsonify({
                "error": "No valid folder structure found in ZIP",
                "required_structure": {
                    "description": "ZIP file must contain a main folder with timestamp subfolders",
                    "example": {
                        "zip_contents": [
                            "myceliumtest1/",
                            "myceliumtest1/24-11-12___18-59/",
                            "myceliumtest1/24-11-12___18-59/1.jpg",
                            "myceliumtest1/24-11-12___18-59/2.jpg",
                            "myceliumtest1/24-11-13___19-30/",
                            "myceliumtest1/24-11-13___19-30/1.jpg",
                            "myceliumtest1/24-11-13___19-30/2.jpg"
                        ]
                    },
                    "folder_format": "YY-MM-DD___HH-MM (e.g., 24-11-12___18-59)",
                    "image_format": "1.jpg, 2.jpg, 3.jpg, 4.jpg (for different angles)",
                    "main_folder": "Can be named anything (e.g., myceliumtest1, experiment_data, etc.)"
                }
            }), 400
        
        job_id = str(uuid.uuid4())
        
        training_jobs[job_id] = {
            "status": "started",
            "created_at": datetime.now().isoformat(),
            "message": "Training job queued...",
            "temp_dir": temp_dir
        }
        
        thread = threading.Thread(target=background_training, args=(job_id, main_folder))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "status": "started",
            "job_id": job_id,
            "message": "Training started in background."
        }), 202
        
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify({"error": f"Failed to process ZIP: {str(e)}"}), 500

@app.route("/retrain/status/<job_id>", methods=["GET"])
def get_training_status(job_id):
    """Check training job status"""
    if job_id not in training_jobs:
        return jsonify({"error": "Job ID not found"}), 404
    
    job_info = training_jobs[job_id]
    return jsonify(job_info), 200

@app.route("/retrain/jobs", methods=["GET"])
def list_training_jobs():
    """List all training jobs"""
    return jsonify({"jobs": training_jobs}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)