from flask import Flask, request, jsonify, send_file
from random import random as rng
from flask_cors import CORS
from segmentation.segment_mycelium import segment_image
from prediction.predictor import predict_growth_stage
from clustering.clusterer import cluster_image
import io
import sys
import psutil 
import os

app = Flask(__name__)

allowed_origins = [
    "https://koenmw.github.io",
    "http://localhost:5173"
]

CORS(app, origins=allowed_origins)

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
    segmented_image.seek(0)  # Reset stream position    conda activate your_env_name
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

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8000)