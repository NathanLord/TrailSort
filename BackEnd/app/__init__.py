from flask import Flask, after_this_request
from flask_cors import CORS
from app.routes import sort_routes, user_routes
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

    processed_folder = os.getenv("PROCESSED_FOLDER")
    logger.info(f"Processed folder path: {processed_folder}")

    @app.after_request
    def after_request(response):
        # Define path to processed zip file
        zip_path = os.path.join(processed_folder, 'processed_images.zip')

        # If the file exists, delete it after sending the response
        if os.path.exists(zip_path):
            try:
                logger.info(f"Deleting the processed zip file: {zip_path}")
                os.remove(zip_path)
            except Exception as e:
                logger.error(f"Error while deleting the zip file: {e}")
        
        return response

    return app
