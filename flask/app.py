from flask import Flask, request, jsonify, send_file
from random import random as rng
from flask_cors import CORS
from segmentor import segment_image

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

    return send_file(segmented_image, mimetype='image/png')

@app.route('/cluster', methods=[ 'POST' ])
def cluster():
	return jsonify({
		"message": "Not yet implemented, random position given",
		"position": {
			"x": rng(),
			"y": rng()
        }
    })	

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8000)