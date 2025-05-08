from flask import Flask
from api.routers import api_blueprint

app = Flask(__name__)

# Register the blueprint
app.register_blueprint(api_blueprint)

if __name__ == '__main__':
    app.run(debug=True)