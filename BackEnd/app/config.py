import os

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
PROCESSED_FOLDER = os.path.join(os.getcwd(), 'processed')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

class Config:
    UPLOAD_FOLDER = UPLOAD_FOLDER
    PROCESSED_FOLDER = PROCESSED_FOLDER
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit
