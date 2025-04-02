import logging

import psycopg2
from psycopg2 import sql
from app.extensions import db  # db in extensions
from sqlalchemy.engine.url import make_url
import json
import base64


import jwt
from datetime import datetime, timedelta, timezone

from app.config import Config  
JWT_SECRET_KEY = Config.JWT_SECRET_KEY


logger = logging.getLogger(__name__)


def retrieve_blogs():

    conn = None # conn is the connection to the db
    cur = None # cur is the cursor which is based off of the connection. It alllows you to execute sql commands

    try:

        database_url_obj = make_url(db.engine.url) 
        database_url = f"postgresql://{database_url_obj.username}:{database_url_obj.password}@{database_url_obj.host}:{database_url_obj.port}/{database_url_obj.database}"

        conn = psycopg2.connect(database_url)
        cur = conn.cursor()

        query_select = "SELECT id, title, content, author, date, image FROM blog_posts"
        
        # Execute the query with the blog content and title
        cur.execute(query_select)  # Pass the title and blog_content as a tupleert, 
   
        # Fetch the results
        blog_posts = cur.fetchall()  # This will return a list of tuples

        # Prepare a list of dictionaries to send to the front-end
        blog_data = []
        for post in blog_posts:
            post_id, title, content, author, date, image = post  # Each post is a tuple (title, content, author, date, image)

            # If there is an image, convert it to base64 (you can adjust this based on your needs)
            if image:
                image_base64 = base64.b64encode(image).decode('utf-8')  # Convert the binary image to base64 string
            else:
                image_base64 = None  # Handle case where no image is provided

            blog_data.append({
                "id": post_id,
                "title": title,
                "content": content,
                "author": author,
                "date": date,
                "image": image_base64  # Include the image as base64 string (or as URL if needed)
            })

        # Convert blog_data to JSON (if using Flask, for example)
        return json.dumps(blog_data)  # This can be used as the response in an API

    except Exception as e:
        logger.error(f"Error while retrieving blog posts: {e}")
        return {"error": "Unable to fetch blog posts."}

    finally:
        # Ensure the cursor and connection are closed
        if cur:
            cur.close()
        if conn:
            conn.close()


def retrieve_blog_post(id):
    conn = None # conn is the connection to the db
    cur = None # cur is the cursor which is based off of the connection. It alllows you to execute sql commands

    try:

        database_url_obj = make_url(db.engine.url) 
        database_url = f"postgresql://{database_url_obj.username}:{database_url_obj.password}@{database_url_obj.host}:{database_url_obj.port}/{database_url_obj.database}"

        conn = psycopg2.connect(database_url)
        cur = conn.cursor()

        query_select = "SELECT id, title, content, author, date, image FROM blog_posts WHERE id = %s"
        
        # Execute the query with the blog content and title
        cur.execute(query_select, (id,))  # Pass the title and blog_content as a tupleert, 
        blog_post = cur.fetchone()  # This will return a list of tuples

        # Prepare a list of dictionaries to send to the front-end
        if blog_post:
            post_id, title, content, author, date, image = blog_post  # Each post is a tuple (title, content, author, date, image)

            # If there is an image, convert it to base64 (you can adjust this based on your needs)    
            if image:
                image_base64 = base64.b64encode(image).decode('utf-8')  # Convert the binary image to base64 string
            else:
                image_base64 = None  # Handle case where no image is provided

            blog_data = {
                "id": post_id,
                "title": title,
                "content": content,
                "author": author,
                "date": date,
                "image": image_base64  # Include the image as base64 string (or as URL if needed)
            }

            # Convert blog_data to JSON (if using Flask, for example)
            return json.dumps(blog_data)  # This can be used as the response in an API

        else:
            return {"error": "Blog post not found."}

    except Exception as e:
        logger.error(f"Error while retrieving blog posts: {e}")
        return {"error": "Unable to fetch blog posts."}

    finally:
        # Ensure the cursor and connection are closed
        if cur:
            cur.close()
        if conn:
            conn.close()    
