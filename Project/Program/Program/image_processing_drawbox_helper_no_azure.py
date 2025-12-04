from tkinter import font
import requests
from PIL import Image, ImageDraw, ImageFont
import os
import json
import base64
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('image_processing.log'),
        logging.StreamHandler()
    ]
)

# ==================== Configuration ====================
BASE_URL = "http://www.prom.ddnsfree.com:8989"  # Replace with your actual base URL
INPUT_FOLDER = "input_images"
FONT_PATH = "AI_Model/Fonts/THSarabunNew BoldItalic.ttf"
OUTPUT_FOLDER = "labelme_output"

username = "helloworld"  # Replace with your actual username
password = "6540039"   # Replace with your actual password
key = "4vWUnXOgmI9Sb7VAeyjEYP3T8divCPJYtYZgmYYDeUxXhpJTetIcJQQJ99BBACYeBjFXJ3w3AAALACOGeLef"             # Replace with your actual key
endpoint = "https://dicon.cognitiveservices.azure.com/"  # Replace with your actual endpoint
type_doc = 2  # 1 for partial bills, 2 for completed bills

type_dict = {1: "prebuilt-receipt", 2: "prebuilt-invoice"}

# Dictionary to store a unique color for each class
class_colors = {}

def get_color_for_class(class_name):
    """Return a fixed color for the given class name."""
    if class_name in class_colors:
        return class_colors[class_name]
    else:
        color_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        assigned_color = color_palette[len(class_colors) % len(color_palette)]
        class_colors[class_name] = assigned_color
        return assigned_color

def convert_image_to_rgb(image_path):
    """
    Convert image to RGB mode, handling various image modes.
    Returns the path to the converted image.
    """
    try:
        # Open the original image
        original_image = Image.open(image_path)
        
        # Create output folder if it doesn't exist
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        # Generate a unique filename for the converted image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        converted_image_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_converted.jpg")
        
        # Convert image to RGB
        original_image.convert('RGB').save(converted_image_path, 'JPEG')
        
        return converted_image_path
    except Exception as e:
        logging.error(f"Error converting image {image_path}: {e}")
        return None

def process_image(image_path, prediction_url, headers, font, datas):
    """Process the image and make predictions using the remote service."""
    logging.info(f"Processing image: {image_path}")
    
    # Convert the image to RGB
    rgb_image_path = convert_image_to_rgb(image_path)
    if rgb_image_path is None:
        logging.error(f"Skipping image due to conversion error: {image_path}")
        return

    # Prepare the image for sending to the prediction service
    with open(rgb_image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
        
    payload = {
        "data": {
            "image": encoded_image,
            "key": datas[0],
            "endpoint": datas[1],
            "type": datas[2],
        }
    }

    # Send the image for prediction
    try:
        response = requests.post(prediction_url, headers=headers, json=payload)
        response.raise_for_status()
        predictions = response.json()
        
        # Handle predictions (this part can be customized based on your needs)
        draw_predictions(rgb_image_path, predictions)
        
    except requests.RequestException as e:
        logging.error(f"Error during prediction for {image_path}: {e}")

def draw_predictions(image_path, predictions):
    """Draw bounding boxes and labels on the image based on predictions."""
    try:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        for item in predictions.get("predictions", []):
            class_name = item["class"]
            confidence = item["confidence"]
            bbox = item["bounding_box"]  # Assuming the bounding box is provided in the predictions
            
            # Draw bounding box
            color = get_color_for_class(class_name)
            draw.rectangle(bbox, outline=color, width=2)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            draw.text((bbox[0], bbox[1]), label, fill=color, font=font)

        # Save the annotated image
        output_image_path = os.path.join(OUTPUT_FOLDER, os.path.basename(image_path))
        image.save(output_image_path)
        logging.info(f"Annotated image saved: {output_image_path}")
        
    except Exception as e:
        logging.error(f"Error drawing predictions on {image_path}: {e}")

def main():
    # Login and get token
    login_url = f"{BASE_URL}/login"
    credentials = {"username": username, "password": password}
    
    logging.info("=== Logging in ===")
    try:
        resp = requests.post(login_url, json=credentials)
        resp.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Login failed: {e}")
        return
    
    access_token = resp.json().get("access_token")
    if not access_token:
        logging.error("Access token not found in response.")
        return
    logging.info("Login successful!")
    
    # Prepare output folder and prediction URL
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    prediction_url = f"{BASE_URL}/predictionboxMS"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # Load the custom font
    try:
        font = ImageFont.truetype(FONT_PATH, size=16)
    except IOError:
        logging.warning("Custom font not found, using default.")
        font = ImageFont.load_default()
    
    # Find image files
    if not os.path.exists(INPUT_FOLDER):
        logging.error(f"Input folder '{INPUT_FOLDER}' does not exist.")
        return

    allowed_extensions = (".png", ".jpg", ".jpeg")
    image_files = []
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for file in files:
            if file.lower().endswith(allowed_extensions):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        logging.error(f"No image files found in folder '{INPUT_FOLDER}'.")
        return

    logging.info(f"Found {len(image_files)} image files to process.")
    
    # Track processing progress
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    datas = [key, endpoint, type_dict[type_doc]]
    
    # Process images
    for image_path in image_files:
        try:
            process_image(image_path, prediction_url, headers, font, datas)
            processed_count += 1
            logging.info(f"Progress: {processed_count}/{len(image_files)} images processed")
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            error_count += 1
    
    # Final summary
    logging.info("\n--- Processing Summary ---")
    logging.info(f"Total images: {len(image_files)}")
    logging.info(f"Successfully processed: {processed_count}")
    logging.info(f"Skipped images: {skipped_count}")
    logging.info(f"Errors encountered: {error_count}")

if __name__ == "__main__":
    main()
