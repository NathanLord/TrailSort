from flask import Blueprint, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import logging
from app.config import UPLOAD_FOLDER
from app.controllers.blog_editor_publish import publish_blog   
from app.controllers.blog_retrieve import retrieve_blogs, retrieve_blog_post

bp = Blueprint('blog_routes', __name__)

logger = logging.getLogger(__name__)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/blog/editor', methods=['POST'])
def publish_blog_post():
        try:

                # Handle multipart form data
                title = request.form.get('title')
                content = request.form.get('content')
                author = request.form.get('author')
                date = request.form.get('date')
                
                # Process the image
                image_file = request.files['image']

                if image_file and allowed_file(image_file.filename):
                    # Read the image as binary
                    image_binary = image_file.read()

                    # Pass the blog data and image binary to the controller
                    publish_blog(title, content, author, date, image_binary)

                    return jsonify({
                        'message': 'Blog post published successfully with image!',
                        'data': {
                            'title': title,
                            'author': author,
                            'date': date,
                        }
                    }), 200
                else:
                    return jsonify({'error': 'Invalid image file type.'}), 400

        except Exception as e:
            logger.error(f"Error while processing the request: {e}", exc_info=True)
            return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500
    

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
    

@bp.route('/blog/<int:blog_id>', methods=['GET'])
def get_blog_by_id(blog_id):
    try:
        # Call retrieve_blogs to get the blog data
        blog_post = retrieve_blog_post(blog_id)

        # If retrieve_blogs returns an error or empty, handle it
        if not blog_post:
            return jsonify({'message': 'No blog post found.'}), 404

        # Return the blog posts as JSON
        return blog_post  # Returns the data as a JSON response
        
    except Exception as e:
        logger.error(f"Error while processing the request: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500


        