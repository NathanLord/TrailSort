from flask import Blueprint, request, jsonify, send_file
from app.controllers.sort_controller import process_file_upload
import logging

bp = Blueprint('sort_routes', __name__)

logger = logging.getLogger(__name__)

@bp.route('/sort', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        result_path = process_file_upload(file)
        return send_file(result_path, as_attachment=True)

    except Exception as e:
        logger.error(f"Error during file upload: {e}")
        return jsonify({"error": str(e)}), 500
