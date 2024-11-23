import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
PROCESSED_FOLDER = os.path.join(os.getcwd(), 'processed')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

class Config:
    UPLOAD_FOLDER = UPLOAD_FOLDER
    PROCESSED_FOLDER = PROCESSED_FOLDER
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit
    SQLALCHEMY_TRACK_MODIFICATIONS = False  
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")
    JWT_SECRET_KEY = os.getenv("SECRET_KEY")
