from flask import Blueprint, request, jsonify
from services.split_data import split_mycelium_by_hours
from services.prediction_model import load_pretrained_model
from services.cluster_mycelium import process_images

api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/split', methods=['POST'])
def split_files():
    data = request.json
    source_folder = data.get('source_folder', 'mycelium')
    output_folder = data.get('output_folder', 'mycelium_labeled')
    tests_to_include = data.get('tests_to_include', [2, 4, 5, 6, 7, 8])
    hour_split = data.get('hour_split', 24)

    split_mycelium_by_hours(
        source_folder=source_folder,
        output_folder=output_folder,
        tests_to_include=tests_to_include,
        hour_split=hour_split
    )
    return jsonify({"message": "Files split successfully!"})

@api_blueprint.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_name = data.get('model_name', 'vgg16')
    model = load_pretrained_model(model_name)
    # Add logic to use the model for predictions
    return jsonify({"message": f"Model {model_name} loaded successfully!"})

@api_blueprint.route('/cluster', methods=['POST'])
def cluster():
    data = request.json
    process_images(
        model_name=data.get('model_name', 'vgg16'),
        angles=data.get('angles', [1]),
        tests=data.get('tests', list(range(1, 7))),
        time_window=data.get('time_window', 2)
    )
    return jsonify({"message": "Clustering completed successfully!"})