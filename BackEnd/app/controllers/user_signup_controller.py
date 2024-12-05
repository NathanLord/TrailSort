import logging

import psycopg2
from psycopg2 import sql
from app.extensions import db  # db in extensions
from sqlalchemy.engine.url import make_url

from werkzeug.security import generate_password_hash


logger = logging.getLogger(__name__)

# https://www.geeksforgeeks.org/making-a-flask-app-using-a-postgresql-database/
# https://www.geeksforgeeks.org/using-jwt-for-user-authentication-in-flask/

def signup_user(username, password):

    conn = None # conn is the connection to the db
    cur = None # cur is the cursor which is based off of the connection. It alllows you to execute sql commands

    try:

        if not username or not password:
            raise ValueError("Username and password are required.")
        
        hashed_password = generate_password_hash(password)

        # The environment password would continue to fail even though there is no difference to the hardcoded one and the environment one.
        #database_url = str(db.engine.url)
        #print(f"Database URL: {database_url}")

        # work around to problem with environment password failing when passed as a str
        # https://docs.sqlalchemy.org/en/20/core/engines.html
        database_url_obj = make_url(db.engine.url) 
        database_url = f"postgresql://{database_url_obj.username}:{database_url_obj.password}@{database_url_obj.host}:{database_url_obj.port}/{database_url_obj.database}"

        conn = psycopg2.connect(database_url)
        cur = conn.cursor()

        # Check db for user
        query_check = sql.SQL("SELECT * FROM users WHERE username = %s")
        cur.execute(query_check, (username,))
        result = cur.fetchone()

        if result:
            raise ValueError("Username already exists")

        # Insert
        query_insert = sql.SQL("INSERT INTO users (username, password) VALUES (%s, %s)")
        cur.execute(query_insert, (username, hashed_password))

        conn.commit()  # Save

        return {"message": "User created successfully", "username": username}
    
    except ValueError as ve:
        logger.error(f"Signup error: {ve}")
        return {"error": str(ve)} 
    
    except Exception as e:
        if conn:
            conn.rollback()  # Revert changes in case of error
        logger.error(f"Error during user signup: {e}")
        return {"error": str(e)}, 500
    
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
