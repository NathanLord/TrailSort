from flask import Blueprint, request, jsonify, send_file

import logging

from app.controllers.sort_controller import process_file_upload
from app.utils.jwt_decorator import token_required

bp = Blueprint('sort_routes', __name__)

logger = logging.getLogger(__name__)

@bp.route('/sort', methods=['POST'])
@token_required
def upload_file(public_id):

    try:

        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        # Get the uploaded file
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Check if the model is part of form
        model_type = request.form.get('model_type')
        if not model_type:
            return jsonify({"error": "Model not selected"}), 400

        # Logic to send to the front end before deleting last zip file we made
        result_path = process_file_upload(file, model_type)
        sorted_folder = send_file(result_path, as_attachment=True)
        
        return sorted_folder

    except Exception as e:
        logger.error(f"Error during file upload: {e}")
        return jsonify({"error": str(e)}), 500
