from flask import Blueprint, request, jsonify
import os
import logging
from app.controllers.user_signup_controller import signup_user

bp = Blueprint('user_routes', __name__)

logger = logging.getLogger(__name__)

@bp.route('/user/signup', methods=['POST'])
def user_signup():
    try:

        username = request.form.get('username')
        password = request.form.get('password')

        logger.info(f"Received form data: username={username}, password={password}")

        web_token = signup_user(username, password)
        # Return the web token (or any other response you need)
        return jsonify({"token": web_token}), 200

    except Exception as e:
        logger.error(f"Error during user signup: {e}")
        return jsonify({"error": str(e)}), 500