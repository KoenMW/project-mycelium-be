from flask import Flask, request, jsonify
from random import random as rng

app = Flask(__name__)

@app.route('/')
def hello():
	return "Server is ok!"

@app.route('/predict', methods = [ 'POST' ])
def predict():
    data = request.get_json()
    print(data)
    day = round(rng() * 20)
    # Add logic to use the model for predictions
    return jsonify({"message": "Not yet implemented, random day given",
    "day": day})

@app.route('/cluster', methods=[ 'POST' ])
def cluster():
	data = request.get_json()
	print(data)
	return jsonify({
		"message": "Not yet implemented, random position given",
		"position": {
			"x": rng(),
			"y": rng()
        }
    })	

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8000)