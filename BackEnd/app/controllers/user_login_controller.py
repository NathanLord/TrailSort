import logging

import psycopg2
from psycopg2 import sql
from app.extensions import db  # db in extensions
from sqlalchemy.engine.url import make_url

from werkzeug.security import check_password_hash
import jwt
from datetime import datetime, timedelta, timezone

from app.config import Config  
JWT_SECRET_KEY = Config.JWT_SECRET_KEY


logger = logging.getLogger(__name__)

# https://www.geeksforgeeks.org/making-a-flask-app-using-a-postgresql-database/
# https://www.geeksforgeeks.org/using-jwt-for-user-authentication-in-flask/ 

def login_user(username, password):

    conn = None # conn is the connection to the db
    cur = None # cur is the cursor which is based off of the connection. It alllows you to execute sql commands

    try:

        if not username or not password:
            raise ValueError("Username and password are required.")
        
        # work around
        # https://docs.sqlalchemy.org/en/20/core/engines.html
        database_url_obj = make_url(db.engine.url) 
        database_url = f"postgresql://{database_url_obj.username}:{database_url_obj.password}@{database_url_obj.host}:{database_url_obj.port}/{database_url_obj.database}"
        

        conn = psycopg2.connect(database_url)
        cur = conn.cursor()

        # Check db for user
        #query_check = sql.SQL("SELECT * FROM users WHERE username = %s")
        query_check = sql.SQL("SELECT id, username, password, first_name, last_name, role FROM users WHERE username = %s")
        cur.execute(query_check, (username,))
        user = cur.fetchone()

        if not user:
            raise ValueError("User does not exist")

        stored_password_hash = user[2]  # third column in db is password
        first_name = user[3]  # fourth column in db is first_name
        last_name = user[4]  # fifth column in db is last_name
        role = user[5]

        # display first and last name and role
        print(f"First name: {first_name}, Last name: {last_name}, Role: {role}")
        
        # Check if the provided password matches the stored hash
        if check_password_hash(stored_password_hash, password):
            token = jwt.encode({
                'public_id': user[0],  # UserID is the first column in db
                'exp': datetime.now(timezone.utc) + timedelta(minutes=30)  # Expiration time
            }, JWT_SECRET_KEY, algorithm='HS256')  

            return {
                'token': token, 
                'first_name': first_name,
                'last_name': last_name,
                'role': role
                }

        else:

            return {'error': 'Invalid password'}
        
    except ValueError as ve:
        logger.error(f"Login error: {ve}")
        return {"error": str(ve)}  
    
    except Exception as e:
        logger.error(f"Login error: {e}")
        return {"error": str(e)}, 500  

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

