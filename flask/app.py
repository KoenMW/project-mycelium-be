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
app = Flask(__name__)

allowed_origins = [
    "https://koenmw.github.io",
    "http://localhost:5173"
]

CORS(app, origins=allowed_origins)

# Global dictionary to track training jobs
training_jobs = {}

def background_training(job_id, upload_folder, quick_mode=False):
    """Background training function"""
    try:
        print(f"🚀 Starting background training job {job_id}")
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["message"] = "Training in progress..."
        
        version = full_retrain_pipeline(upload_folder, quick_mode=quick_mode)
        
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["version"] = version
        training_jobs[job_id]["message"] = f"Training completed successfully! Version: {version}"
        print(f"✅ Background training job {job_id} completed")
        
    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)
        training_jobs[job_id]["message"] = f"Training failed: {str(e)}"
        print(f"❌ Background training job {job_id} failed: {str(e)}")

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

    # Perform segmentation
    segmented_image, error = segment_image(file.read())
    if error:
        return jsonify({ "error": error }), 400
    
    # === Prediction ===
    segmented_image.seek(0)  # Reset stream position
    prediction_result, prediction_error = predict_growth_stage(segmented_image.read())
    if prediction_error:
        return jsonify({ "error": prediction_error }), 500

    return jsonify({
        "prediction": prediction_result
    })

@app.route('/cluster', methods=['POST'])
def cluster():
    if 'file' not in request.files:
        return jsonify({ "error": "No file uploaded" }), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({ "error": "Empty filename" }), 400

    # Optional: extract hour (if provided by client)
    hour = request.form.get("hour")
    try:
        hour = int(hour) if hour else None
    except ValueError:
        return jsonify({ "error": "Invalid hour format" }), 400

    result, error = cluster_image(io.BytesIO(file.read()), hour)
    if error:
        return jsonify({ "error": error }), 500

    return jsonify({ "clustering": result })

@app.route('/health')
def health():
    # Check model files
    model_files = [
        "../models/best_hybrid_model.keras",
        "../models/encoder_model.keras",
        "../models/hdbscan_clusterer.pkl",
        "../models/pca_model.pkl",
        "../models/yolo_segmenting_model.pt"
    ]
    missing = [f for f in model_files if not os.path.exists(f)]
    status = "ok" if not missing else "degraded"

    # Optionally, add resource info
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss // (1024 * 1024)  # MB

    return jsonify({
        "status": status,
        "missing_files": missing,
        "python_version": sys.version,
        "memory_usage_mb": mem
    }), 200 if status == "ok" else 503

@app.route('/metadata')
def metadata():
    return jsonify({
        "project": "mycelium-be",
        "version": "1.0.0",
        "author": "Group 6",
        "description": "API for mycelium project"
    }), 200

@app.route("/retrain", methods=["POST"])
def retrain_models():
    """Start training in background and return job ID"""
    data = request.get_json()
    upload_folder = data.get("upload_folder")

    if not upload_folder or not os.path.exists(upload_folder):
        return jsonify({"error": f"Invalid or missing folder: {upload_folder}"}), 400

    # Create unique job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job tracking
    training_jobs[job_id] = {
        "status": "started",
        "created_at": datetime.now().isoformat(),
        "message": "Training job queued...",
        "quick_mode": False
    }
    
    # Start background thread
    thread = threading.Thread(target=background_training, args=(job_id, upload_folder, False))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "status": "started",
        "job_id": job_id,
        "message": "Training started in background. Use /retrain/status/{job_id} to check progress."
    }), 202

@app.route("/retrain-quick", methods=["POST"])
def retrain_quick():
    """Start quick training in background and return job ID"""
    data = request.get_json()
    upload_folder = data.get("upload_folder")

    if not upload_folder or not os.path.exists(upload_folder):
        return jsonify({"error": f"Invalid or missing folder: {upload_folder}"}), 400

    # Create unique job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job tracking
    training_jobs[job_id] = {
        "status": "started",
        "created_at": datetime.now().isoformat(),
        "message": "Quick training job queued...",
        "quick_mode": True
    }
    
    # Start background thread
    thread = threading.Thread(target=background_training, args=(job_id, upload_folder, True))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "status": "started",
        "job_id": job_id,
        "message": "Quick training started in background. Use /retrain/status/{job_id} to check progress."
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)