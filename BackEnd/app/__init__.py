from flask import Flask
from flask_cors import CORS
from app.routes import sort_routes, user_routes, blog_routes
import os
import logging
from dotenv import load_dotenv
from app.extensions import db
from app.config import Config

load_dotenv()


def create_app():
    
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})

    app.config.from_object(Config)

    # Start db
    db.init_app(app)

    logger = logging.getLogger(__name__)

    # Add blueprints for different routes
    app.register_blueprint(sort_routes.bp)
    app.register_blueprint(user_routes.bp)
    app.register_blueprint(blog_routes.bp)

    processed_folder = os.getenv("PROCESSED_FOLDER")
    logger.info(f"Processed folder path: {processed_folder}")


    return app
