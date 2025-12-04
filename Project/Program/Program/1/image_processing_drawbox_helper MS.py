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
BASE_URL = "http://www.prom.ddnsfree.com:8989"
INPUT_FOLDER = "input_images"
FONT_PATH = "AI_Model/Fonts/THSarabunNew BoldItalic.ttf"
OUTPUT_FOLDER = "labelme_output"

username = "helloworld"
password = "6540039"
key = "4vWUnXOgmI9Sb7VAeyjEYP3T8divCPJYtYZgmYYDeUxXhpJTetIcJQQJ99BBACYeBjFXJ3w3AAALACOGeLef"
endpoint = "https://dicon.cognitiveservices.azure.com/"
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
        if original_image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', original_image.size, (255, 255, 255))
            background.paste(original_image, mask=original_image.split()[-1])
            background.save(converted_image_path, 'JPEG')
        elif original_image.mode != 'RGB':
            original_image.convert('RGB').save(converted_image_path, 'JPEG')
        else:
            original_image.save(converted_image_path, 'JPEG')
        
        return converted_image_path
    except Exception as e:
        logging.error(f"Error converting image {image_path}: {e}")
        return None

def process_image(image_path, prediction_url, headers, font, datas):
    """
    Process a single image: send it for prediction, draw annotations,
    generate LabelMe JSON, and save outputs.
    """
    logging.info(f"Processing image: {image_path}")
    
    # Convert image to RGB if needed
    converted_image_path = convert_image_to_rgb(image_path)
    if not converted_image_path:
        logging.error(f"Failed to convert image: {image_path}")
        return
    
    # Prepare data for prediction
    data = {
        "key": datas[0],
        "endpoint": datas[1],
        "type_doc": datas[2]
    }
    
    # Send image for prediction
    try:
        with open(converted_image_path, "rb") as f:
            files = {"image": (os.path.basename(converted_image_path), f, "image/jpeg")}
            resp = requests.post(prediction_url, files=files, data=data, headers=headers)
    except Exception as e:
        logging.error(f"Prediction request failed for {image_path}: {e}")
        return
    
    # Check prediction response
    if resp.status_code != 201:
        logging.error(f"Prediction failed for {image_path}: {resp.text}")
        return
    
    # Parse predictions
    try:
        predictions = resp.json().get("results", [])
        logging.info(f"Received {len(predictions)} predictions for {os.path.basename(image_path)}")
    except Exception as e:
        logging.error(f"Failed to parse predictions for {image_path}: {e}")
        return
    
    # Open the converted image for drawing
    image = Image.open(converted_image_path)
    draw = ImageDraw.Draw(image)
    shapes = []
    
    # Process each prediction
    for prediction_set in predictions:
        documents = prediction_set.get('documents', [])
        
        for document in documents:
            fields = document.get('fields', {})
            
            def process_field(field_name, field, parent_name=""):
                # Handle array-type fields
                if field.get('type') == 'array':
                    for idx, item in enumerate(field.get('value_array') or []):
                        if 'value_object' in item:
                            for sub_field_name, sub_field in item['value_object'].items():
                                full_name = f"{sub_field_name}"
                                process_field(full_name, sub_field)
                    return
                
                # Handle bounding regions
                bounding_regions = field.get('bounding_regions') or []
                
                for region in bounding_regions:
                    polygon = region.get('polygon', [])
                    if len(polygon) != 8:
                        continue

                    x_coords = polygon[::2]
                    y_coords = polygon[1::2]
                    
                    x_min = min(x_coords)
                    x_max = max(x_coords)
                    y_min = min(y_coords)
                    y_max = max(y_coords)
                    
                    confidence = field.get('confidence', 0.0)
                    final_field_name = parent_name + field_name

                    # Get class color
                    color = get_color_for_class(final_field_name)

                    # Draw bounding box
                    draw.rectangle([(x_min, y_min), (x_max, y_max)], 
                                 outline=color, width=2)

                    # Draw label
                    label_text = f"{final_field_name}: {confidence:.2f}"
                    draw.text((x_min, y_min - 20), label_text, 
                             font=font, fill=color)

                    # Add to shapes
                    shapes.append({
                        "label": final_field_name,
                        "points": [[x_min, y_min], [x_max, y_max]],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {}
                    })
            
            # Process all top-level fields
            for field_name, field in fields.items():
                process_field(field_name, field)
    
    # Save annotated image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_image_path = os.path.join(OUTPUT_FOLDER, base_name + "_prediction.jpg")
    image.save(output_image_path)
    logging.info(f"Saved preview image with predictions to: {output_image_path}")
    
    # Generate and save LabelMe JSON
    width, height = image.size
    labelme_json = generate_labelme_json(image_path, shapes, width, height)
    json_filename = os.path.join(OUTPUT_FOLDER, base_name + ".json")
    with open(json_filename, "w") as jf:
        json.dump(labelme_json, jf, indent=4)
    logging.info(f"Saved LabelMe annotation JSON to: {json_filename}")
    
    # Copy original image
    image_copy_path = os.path.join(OUTPUT_FOLDER, os.path.basename(image_path))
    shutil.copy(image_path, image_copy_path)
    logging.info(f"Copied original image to: {image_copy_path}")

def generate_labelme_json(image_path, shapes, image_width, image_height):
    """Create a LabelMe-style JSON annotation."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
        
    labelme_json = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": image_data,
        "imageWidth": image_width,
        "imageHeight": image_height
    }
    return labelme_json

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
