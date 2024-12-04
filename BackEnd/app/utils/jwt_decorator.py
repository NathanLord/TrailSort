from functools import wraps
from flask import request, jsonify
import jwt
from app.config import Config  
JWT_SECRET_KEY = Config.JWT_SECRET_KEY

# https://www.geeksforgeeks.org/using-jwt-for-user-authentication-in-flask/
# Decorator to require a valid JWT token
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            # Check if the header starts with 'Bearer ' to get the token
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]  # Remove 'Bearer ' part to get the token
            else:
                return jsonify({'message': 'Token is missing or malformed. Try logining in'}), 401

        # Token is not passed
        if not token:
            return jsonify({'message': 'Token is missing! Try logining in.'}), 401

        try:
            # Decode the token using the secret key
            data = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
            public_id = data['public_id']  # public_id is an argument


        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired! Logout and Log back in.'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid!'}), 401
        except Exception as e:
            return jsonify({'message': str(e)}), 401

        # Pass object to the route function
        return f(public_id, *args, **kwargs)

    return decorated
