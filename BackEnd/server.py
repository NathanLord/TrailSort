




import os
import shutil
import zipfile
from pathlib import Path
import uuid
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

# Load your model
model = load_model("model/trailSortTF2.keras")

# Define image dimensions
img_height = 180
img_width = 180
target_size = (img_height, img_width)

# Define paths for temporary uploads and output folders
TEMP_UPLOAD_PATH = Path("temp_uploads")
SORTED_OUTPUT_PATH = Path("sorted_outputs")

# Ensure directories exist with proper permissions
def ensure_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o777)  # Full permissions
    logger.debug(f"Created directory {path} with full permissions")

ensure_directory(TEMP_UPLOAD_PATH)
ensure_directory(SORTED_OUTPUT_PATH)

# Define class names based on your model's output
class_names = ["blackBear", "coyote", "ruffedGrouse", "turkey", "whitetailDeer"]

def load_and_preprocess_image(img_path, target_size):
    try:
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logger.error(f"Error processing image {img_path}: {str(e)}")
        raise

def classify_and_sort_images(image_folder, sorted_folder):
    try:
        # Handle the case where the upload is a single file
        if os.path.isfile(image_folder):
            logger.debug(f"Processing single file: {image_folder}")
            img_array = load_and_preprocess_image(image_folder, target_size)
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            predicted_class = class_names[np.argmax(score)]
            
            class_folder = sorted_folder / predicted_class
            ensure_directory(class_folder)
            shutil.copy2(image_folder, class_folder / Path(image_folder).name)
            return

        # Handle directory of images
        logger.debug(f"Processing directory: {image_folder}")
        for item in os.listdir(image_folder):
            item_path = Path(image_folder) / item
            if item_path.is_file() and item.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_array = load_and_preprocess_image(item_path, target_size)
                    predictions = model.predict(img_array)
                    score = tf.nn.softmax(predictions[0])
                    predicted_class = class_names[np.argmax(score)]
                    
                    class_folder = sorted_folder / predicted_class
                    ensure_directory(class_folder)
                    shutil.copy2(item_path, class_folder / item)
                except Exception as e:
                    logger.error(f"Error processing {item}: {str(e)}")
                    continue
    except Exception as e:
        logger.error(f"Error in classify_and_sort_images: {str(e)}")
        raise

@app.route("/sort", methods=["POST"])
def sort_images():
    try:
        if "folder" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        # Create unique directories for this request
        session_id = str(uuid.uuid4())
        upload_folder = TEMP_UPLOAD_PATH / session_id
        sorted_folder = SORTED_OUTPUT_PATH / session_id
        
        ensure_directory(upload_folder)
        ensure_directory(sorted_folder)
        
        logger.debug(f"Created session directories: {upload_folder}, {sorted_folder}")

        # Save and process the uploaded file
        zip_file = request.files["folder"]
        zip_path = upload_folder / "uploaded_images.zip"
        zip_file.save(zip_path)
        logger.debug(f"Saved uploaded file to {zip_path}")

        # Extract the zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            extract_path = upload_folder / "extracted"
            ensure_directory(extract_path)
            zip_ref.extractall(extract_path)
            logger.debug(f"Extracted zip to {extract_path}")

        # Process the images
        classify_and_sort_images(extract_path, sorted_folder)
        logger.debug("Finished classifying images")

        # Create output zip
        output_zip = sorted_folder.with_suffix(".zip")
        with zipfile.ZipFile(output_zip, "w") as zipf:
            for root, _, files in os.walk(sorted_folder):
                for file in files:
                    file_path = Path(root) / file
                    arc_name = file_path.relative_to(sorted_folder)
                    zipf.write(file_path, arc_name)
        
        logger.debug(f"Created output zip at {output_zip}")

        # Clean up temporary files
        shutil.rmtree(upload_folder, ignore_errors=True)
        
        return send_file(output_zip, as_attachment=True)

    except Exception as e:
        logger.error(f"Error in sort_images endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up sorted folder after sending the file
        shutil.rmtree(sorted_folder, ignore_errors=True)

if __name__ == "__main__":
    app.run(debug=True)


