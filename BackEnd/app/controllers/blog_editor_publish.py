import logging

import psycopg2
from psycopg2 import sql
from app.extensions import db  # db in extensions
from sqlalchemy.engine.url import make_url


import jwt
from datetime import datetime, timedelta, timezone

from app.config import Config  
JWT_SECRET_KEY = Config.JWT_SECRET_KEY


logger = logging.getLogger(__name__)
def publish_blog(title, content, author, date, image_binary):

    conn = None # conn is the connection to the db
    cur = None # cur is the cursor which is based off of the connection. It alllows you to execute sql commands

    try:

        database_url_obj = make_url(db.engine.url) 
        database_url = f"postgresql://{database_url_obj.username}:{database_url_obj.password}@{database_url_obj.host}:{database_url_obj.port}/{database_url_obj.database}"

        conn = psycopg2.connect(database_url)
        cur = conn.cursor()

        query_insert = "INSERT INTO blog_posts (title, content, author, date, image) VALUES (%s, %s, %s, %s, %s)"
        
        # Execute the query with the blog data and image binary
        cur.execute(query_insert, (title, content, author, date, psycopg2.Binary(image_binary)))

        conn.commit()  # Save

        return {"message": "Blog published successfully"}
    
    except ValueError as ve:
        logger.error(f"Blog publish error: {ve}")
        return {"error": str(ve)} 
    
    except Exception as e:
        if conn:
            conn.rollback()  # Revert changes in case of error
        logger.error(f"Error during blog publish: {e}")
        return {"error": str(e)}, 500
    
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

        