from flask import Flask
from flask_cors import CORS
from app.routes import sort_routes

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    app.config.from_object('app.config')

    # Add blueprints for different routes
    app.register_blueprint(sort_routes.bp)

    return app
