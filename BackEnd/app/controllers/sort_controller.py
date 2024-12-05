import os
import zipfile
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from app.config import UPLOAD_FOLDER, PROCESSED_FOLDER
import logging


logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



# Image classes
class_names = ["blackBear", "coyote", "ruffedGrouse", "turkey", "whitetailDeer"]


# Function to sort images based on model predictions
def sort_images(extracted_folder, output_folder, model, target_size):
    # Create subfolders in the processed folder for each class
    for class_name in class_names:
        logger.debug(f"Class name:  {class_name}")
        os.makedirs(os.path.join(output_folder, class_name), exist_ok=True)
        # logger.debug(f"Class name after:  {class_name}")

    # Process each image and predict its class
    for root, _, files in os.walk(extracted_folder):
        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file_name)
                
                # Load the image and prepare it
                img = tf.keras.utils.load_img(img_path, target_size=target_size)
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)  # Create a batch

                # Make predictions
                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])  # Get softmax scores

                # Get the predicted class and confidence
                predicted_class = class_names[np.argmax(score)]
                confidence = 100 * np.max(score)

                # Log outcome
                logger.debug(f"Image: {file_name} -> Predicted Class: {predicted_class} with {confidence:.2f}% confidence")

                # Copy the image to the predicted class folder
                shutil.copy(img_path, os.path.join(output_folder, predicted_class, file_name))



def process_file_upload(file, model_type):

    # Dictionary for front-end to correspond to back-end models
    model_paths = {
        'trailSortTF2.keras': "model/trailSortTF2.keras",
        'trailSortTF3Large.keras': "model/trailSortTF3Large.keras",
        'trailSortTF2MorePixels.keras': "model/trailSortTF2MorePixels.keras"
    }

    # Check if the model_type is in the model_paths dictionary
    if model_type not in model_paths:
        raise ValueError(f"Model type, {model_type}, does not match an existing model.")

    # Load the selected model based on the model_type
    model = load_model(model_paths[model_type])

    # Set target size based selected model
    if model_type == 'trailSortTF2MorePixels.keras':
        img_height = 256
        img_width = 256
    else:
        img_height = 180
        img_width = 180
    target_size = (img_height, img_width)


    zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(zip_path)

    # Extract the zip file https://www.geeksforgeeks.org/unzipping-files-in-python/
    extract_folder = os.path.join(UPLOAD_FOLDER, "extracted")
    os.makedirs(extract_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    # Output folder for processed images
    output_folder = os.path.join(PROCESSED_FOLDER, "processed_images")
    os.makedirs(output_folder, exist_ok=True)

    # Sort images based on TensorFlow model predictions
    sort_images(extract_folder, output_folder, model, target_size)

    # Create a zip file of the sorted images https://stackoverflow.com/questions/1855095/how-to-create-a-zip-archive-of-a-directory
    output_zip_path = os.path.join(PROCESSED_FOLDER, 'processed_images.zip')
    with zipfile.ZipFile(output_zip_path, 'w') as zipf:
        for root, _, files in os.walk(output_folder):
            for file_name in files:
                # Get the path of the file from the output folder
                rel_path = os.path.relpath(os.path.join(root, file_name), output_folder)
                # Write to zip
                zipf.write(os.path.join(root, file_name), arcname=rel_path)

    # Clean up the extracted folder and processed images folder
    shutil.rmtree(extract_folder)
    shutil.rmtree(output_folder)

    # Clean up the uploaded file
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    return output_zip_path
