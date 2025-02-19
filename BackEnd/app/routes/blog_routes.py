from flask import Blueprint, request, jsonify, send_file

import logging

from app.controllers.blog_editor_publish import publish_blog   

bp = Blueprint('blog_routes', __name__)

logger = logging.getLogger(__name__)


@bp.route('/blog/editor', methods=['POST'])
def publish_blog_post():
    try:
        
        model_data = request.get_json()

        
        if not model_data:
            return jsonify({'error': 'No data received'}), 400

        
        # logger.info(f"Received model data: {model_data}")

        # Process the model data (e.g., save it to the database, etc.)
        publish_blog(model_data)
        
        return jsonify({'message': 'Blog post published successfully!', 'data': model_data}), 200

    except Exception as e:
        logger.error(f"Error while processing the request: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500