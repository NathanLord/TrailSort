from flask import Blueprint, request, jsonify, send_file

import logging

from app.controllers.blog_editor_publish import publish_blog   
from app.controllers.blog_retrieve import retrieve_blogs

bp = Blueprint('blog_routes', __name__)

logger = logging.getLogger(__name__)


@bp.route('/blog/editor', methods=['POST'])
def publish_blog_post():
    try:
        
        model_data = request.get_json()

        if not model_data:
            return jsonify({'error': 'No data received'}), 400

        title = model_data.get('title')
        content = model_data.get('content')

        if not title or not content:
            return jsonify({'error': 'No title or content received'}), 400

        # logger.info(f"Received model data: {model_data}")

        # Process the model data (e.g., save it to the database, etc.)
        publish_blog(title, content)
        
        return jsonify({'message': 'Blog post published successfully!', 'data': model_data}), 200

    except Exception as e:
        logger.error(f"Error while processing the request: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500
    

# Get all blog posts json
@bp.route('/blog/retrieve', methods=['GET'])
def download_blog_post():
    try:
        # Call retrieve_blogs to get the blog data
        blog_posts = retrieve_blogs()

        # If retrieve_blogs returns an error or empty, handle it
        if not blog_posts:
            return jsonify({'message': 'No blog posts found.'}), 404

        # Return the blog posts as JSON
        return blog_posts  # Returns the data as a JSON response
        
    except Exception as e:
        logger.error(f"Error while processing the request: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

        