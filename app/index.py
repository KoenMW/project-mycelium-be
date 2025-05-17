from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

def get_time_stamp():
    return datetime.now().isoformat() + 'z'


@app.route("/")
def hello_world():
    return "<p>hello world!</br>from the flask api</p>"

@app.route("/time")
def get_time():
    now = get_time_stamp()
    return { 'time': now }

@app.route('/echo', methods = [ 'post' ])
def echo_message():
    data = request.get_json()
    message = data.get("message")
    return jsonify({
        "message": message,
        "timestamp": get_time_stamp()
    })