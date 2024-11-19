import os
import zipfile
import shutil
from app.config import UPLOAD_FOLDER, PROCESSED_FOLDER

def process_file_upload(file):
    zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(zip_path)

    extract_folder = os.path.join(UPLOAD_FOLDER, "extracted")
    os.makedirs(extract_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    output_folder = os.path.join(PROCESSED_FOLDER, "processed_images")
    os.makedirs(output_folder, exist_ok=True)
    for root, _, files in os.walk(extract_folder):
        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy(os.path.join(root, file_name), output_folder)

    output_zip_path = os.path.join(PROCESSED_FOLDER, 'processed_images.zip')
    with zipfile.ZipFile(output_zip_path, 'w') as zipf:
        for root, _, files in os.walk(output_folder):
            for file_name in files:
                zipf.write(os.path.join(root, file_name), arcname=file_name)

    shutil.rmtree(extract_folder)
    return output_zip_path
