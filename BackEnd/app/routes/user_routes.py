from flask import Blueprint, request, jsonify

import logging

from app.controllers.user_signup_controller import signup_user
from app.controllers.user_login_controller import login_user

bp = Blueprint('user_routes', __name__)

logger = logging.getLogger(__name__)


@bp.route('/user/signup', methods=['POST'])
def user_signup():
    try:

        data = request.get_json() 
        username = data.get('username') 
        password = data.get('password')
        firstName = data.get('firstName')
        lastName = data.get('lastName')
        email = data.get('email')


        # logger.info(f"Received form data: username={username}, password={password}")

        results = signup_user(username, password, firstName, lastName, email)

        if 'error' in results:
            return jsonify({"error": results['error']}), 400
        
        return jsonify({"results": results}), 200

    except Exception as e:
        logger.error(f"Error during user signup: {e}")
        return jsonify({"error": str(e)}), 500
    


@bp.route('/user/login', methods=['POST'])
def user_login():
    try:

        data = request.get_json() 
        username = data.get('username') 
        password = data.get('password')

        # logger.info(f"Received form data: username={username}, password={password}")

        results = login_user(username, password)

        
        if 'error' in results:
            return jsonify({"error": results['error']}), 400

        # Return the token
        return jsonify(results), 200  # This will serialize the dictionary as JSON


    except Exception as e:
        logger.error(f"Error during user login: {e}")
        return jsonify({"error": str(e)}), 500