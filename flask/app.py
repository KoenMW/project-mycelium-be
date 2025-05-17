from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
	return "Hello World!"

@app.route('/cache-me')
def cache():
	return "nginx will cache this response"


@app.route('/flask-health-check')
def flask_health_check():
	return "success"