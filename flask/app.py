from flask import Flask, request, jsonify, send_file
from random import random as rng
from flask_cors import CORS
from segmentation.segment_mycelium import segment_image, segment_and_save
from prediction.predictor import predict_growth_stage
from clustering.clusterer import cluster_image
import io
import sys
import psutil 
import os
import zipfile
import tempfile
import shutil
import threading
import uuid
from datetime import datetime
import glob
from database.data import fetch_pocketbase_data, upload_to_pocketbase
import re

# Import background job functions
from jobs import background_training, background_upload, training_jobs, upload_jobs

app = Flask(__name__)

allowed_origins = [
    "https://koenmw.github.io",
    "http://localhost:5173"
]

CORS(app, origins=allowed_origins)

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
                "parameters": [],
                "description": "Retrain models with database data"
            },
            "upload-data": {
                "method": "POST",
                "parameters": ["zip_file (required)", "trainingData (optional, default: true)"],
                "description": "Upload ZIP file with mycelium data"
            },
            "jobs": {
                "method": "GET",
                "parameters": [],
                "description": "List all jobs (training and upload)"
            }
        }
    }), 200

# === Training Endpoints ===
@app.route("/retrain", methods=["POST"])
def retrain_models():
    """Start retraining using only database data where trainingData=true"""
    job_id = str(uuid.uuid4())
    
    training_jobs[job_id] = {
        "status": "started",
        "created_at": datetime.now().isoformat(),
        "message": "Training job queued..."
    }
    
    thread = threading.Thread(target=background_training, args=(job_id,))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "status": "started",
        "job_id": job_id,
        "message": "Training started using database training data."
    }), 202

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

# === Upload Endpoints ===
@app.route("/upload-data", methods=["POST"])
def upload_data():
    """Start background upload job"""
    if 'zip_file' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zip_file']
    if zip_file.filename == '' or not zip_file.filename.lower().endswith('.zip'):
        return jsonify({"error": "Please upload a ZIP file"}), 400

    # Get trainingData flag from form
    is_training_data = request.form.get('trainingData', 'true').lower() == 'true'
    
    # Create job
    job_id = str(uuid.uuid4())
    upload_id = job_id[:8]
    temp_dir = tempfile.mkdtemp(prefix=f"upload_data_{upload_id}_")
    
    # Save uploaded ZIP
    zip_path = os.path.join(temp_dir, "upload.zip")
    zip_file.save(zip_path)
    
    # Initialize job tracking
    upload_jobs[job_id] = {
        "status": "started",
        "created_at": datetime.now().isoformat(),
        "upload_id": upload_id,
        "training_data": is_training_data,
        "progress": 0,
        "message": "Upload job queued...",
        "total_images": 0,
        "uploaded_count": 0,
        "failed_count": 0
    }
    
    # Start background thread
    thread = threading.Thread(
        target=background_upload, 
        args=(job_id, zip_path, is_training_data, temp_dir)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "status": "started",
        "job_id": job_id,
        "upload_id": upload_id,
        "message": "Upload started in background. Use job_id to track progress."
    }), 202

@app.route("/upload/status/<job_id>", methods=["GET"])
def get_upload_status(job_id):
    """Check upload job status"""
    if job_id not in upload_jobs:
        return jsonify({"error": "Job ID not found"}), 404
    
    job_info = upload_jobs[job_id]
    return jsonify(job_info), 200

@app.route("/upload/jobs", methods=["GET"])
def list_upload_jobs():
    """List all upload jobs"""
    return jsonify({"jobs": upload_jobs}), 200

# === General Jobs Endpoint ===
@app.route("/jobs", methods=["GET"])
def list_all_jobs():
    """List all jobs (training and upload) with unified format"""
    all_jobs = []
    
    # Add training jobs
    for job_id, job_info in training_jobs.items():
        unified_job = {
            "job_id": job_id,
            "job_type": "training",
            "status": job_info.get("status"),
            "created_at": job_info.get("created_at"),
            "message": job_info.get("message"),
            "progress": job_info.get("progress", None),  # Training jobs don't have progress
            "details": {
                "version": job_info.get("version"),
                "version_path": job_info.get("version_path"),
                "error": job_info.get("error")
            }
        }
        all_jobs.append(unified_job)
    
    # Add upload jobs
    for job_id, job_info in upload_jobs.items():
        unified_job = {
            "job_id": job_id,
            "job_type": "upload",
            "status": job_info.get("status"),
            "created_at": job_info.get("created_at"),
            "message": job_info.get("message"),
            "progress": job_info.get("progress"),
            "details": {
                "upload_id": job_info.get("upload_id"),
                "training_data": job_info.get("training_data"),
                "total_images": job_info.get("total_images"),
                "uploaded_count": job_info.get("uploaded_count"),
                "failed_count": job_info.get("failed_count"),
                "test_number": job_info.get("test_number"),
                "error": job_info.get("error")
            }
        }
        all_jobs.append(unified_job)
    
    # Sort by creation time (newest first)
    all_jobs.sort(key=lambda x: x["created_at"], reverse=True)
    
    return jsonify({
        "total_jobs": len(all_jobs),
        "training_jobs": len(training_jobs),
        "upload_jobs": len(upload_jobs),
        "jobs": all_jobs
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)