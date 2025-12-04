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
type_doc = 2  # 2 for completed bills

type_dict = {1: "prebuilt-receipt", 2: "prebuilt-invoice"}

# Required classes with specific colors
REQUIRED_CLASSES = {
    'buyer_name_thai': (255, 0, 0),      # Red
    'buyer_name_eng': (0, 255, 0),       # Green
    'seller_name_thai': (0, 0, 255),     # Blue
    'seller_name_eng': (255, 255, 0),    # Yellow
    'buyer_vat_number': (255, 0, 255),   # Magenta
    'seller_vat_number': (0, 255, 255),  # Cyan
    'total_due_amount': (128, 0, 0),     # Dark Red
}

# Dictionary to store colors for additional classes
class_colors = {}

def get_color_for_class(class_name):
    """Return specific colors for required classes, generate colors for others."""
    if class_name in REQUIRED_CLASSES:
        return REQUIRED_CLASSES[class_name]
    if class_name in class_colors:
        return class_colors[class_name]
    
    color_palette = [
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (64, 0, 0), (0, 64, 0), (0, 0, 64)
    ]
    assigned_color = color_palette[len(class_colors) % len(color_palette)]
    class_colors[class_name] = assigned_color
    return assigned_color

def convert_image_to_rgb(image_path):
    """Convert image to RGB mode, handling various image modes."""
    try:
        original_image = Image.open(image_path)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        converted_image_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_converted.jpg")
        
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

def process_field(field_name, field, draw, font, image, shapes):
    """Process individual fields with confidence thresholds."""
    confidence_threshold = 0.4 if field_name in REQUIRED_CLASSES else 0.6
    
    # Safely get confidence value with default of 0.0 if None
    confidence = field.get('confidence')
    if confidence is None:
        confidence = 0.0
    
    if confidence < confidence_threshold:
        return

    # Handle language-specific fields
    if field_name in ['buyer_name_thai', 'buyer_name_eng', 
                     'seller_name_thai', 'seller_name_eng']:
        if not field.get('value'):
            return

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
        color = get_color_for_class(field_name)

        # Draw bounding box
        draw.rectangle([(x_min, y_min), (x_max, y_max)], 
                      outline=color, width=2)

        # Draw label
        label_text = f"{field_name}: {confidence:.2f}"
        draw.text((x_min, y_min - 20), label_text, 
                  font=font, fill=color)

        # Add to shapes
        shapes.append({
            "label": field_name,
            "points": [[x_min, y_min], [x_max, y_max]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        })

def process_image(image_path, prediction_url, headers, font, datas):
    """Process a single image with enhanced field detection."""
    logging.info(f"Processing image: {image_path}")
    
    converted_image_path = convert_image_to_rgb(image_path)
    if not converted_image_path:
        return
    
    data = {
        "key": datas[0],
        "endpoint": datas[1],
        "type_doc": datas[2]
    }
    
    try:
        with open(converted_image_path, "rb") as f:
            files = {"image": (os.path.basename(converted_image_path), f, "image/jpeg")}
            resp = requests.post(prediction_url, files=files, data=data, headers=headers)
    except Exception as e:
        logging.error(f"Prediction request failed for {image_path}: {e}")
        return
    
    if resp.status_code != 201:
        logging.error(f"Prediction failed for {image_path}: {resp.text}")
        return
    
    try:
        predictions = resp.json().get("results", [])
        logging.info(f"Received {len(predictions)} predictions for {os.path.basename(image_path)}")
    except Exception as e:
        logging.error(f"Failed to parse predictions for {image_path}: {e}")
        return
    
    image = Image.open(converted_image_path)
    draw = ImageDraw.Draw(image)
    shapes = []
    detected_fields = set()
    
    for prediction_set in predictions:
        documents = prediction_set.get('documents', [])
        for document in documents:
            fields = document.get('fields', {})
            for field_name, field in fields.items():
                detected_fields.add(field_name)
                process_field(field_name, field, draw, font, image, shapes)
    
    # Log missing required fields
    required_fields = set(REQUIRED_CLASSES.keys())
    missing_required = required_fields - detected_fields
    if missing_required:
        logging.warning(f"Missing required fields in {image_path}: {missing_required}")
    
    # Log additional detected fields
    additional_fields = detected_fields - required_fields
    if additional_fields:
        logging.info(f"Additional fields detected in {image_path}: {additional_fields}")
    
    # Save outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_image_path = os.path.join(OUTPUT_FOLDER, base_name + "_prediction.jpg")
    image.save(output_image_path)
    
    width, height = image.size
    labelme_json = generate_labelme_json(image_path, shapes, width, height)
    json_filename = os.path.join(OUTPUT_FOLDER, base_name + ".json")
    with open(json_filename, "w") as jf:
        json.dump(labelme_json, jf, indent=4)
    
    image_copy_path = os.path.join(OUTPUT_FOLDER, os.path.basename(image_path))
    shutil.copy(image_path, image_copy_path)

def generate_labelme_json(image_path, shapes, image_width, image_height):
    """Create a LabelMe-style JSON annotation."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
        
    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": image_data,
        "imageWidth": image_width,
        "imageHeight": image_height
    }

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
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    prediction_url = f"{BASE_URL}/predictionboxMS"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    try:
        font = ImageFont.truetype(FONT_PATH, size=16)
    except IOError:
        logging.warning("Custom font not found, using default.")
        font = ImageFont.load_default()
    
    if not os.path.exists(INPUT_FOLDER):
        logging.error(f"Input folder '{INPUT_FOLDER}' does not exist.")
        return

    allowed_extensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
    image_files = [os.path.join(root, f) for root, _, files in os.walk(INPUT_FOLDER)
                   for f in files if f.lower().endswith(allowed_extensions)]
    
    if not image_files:
        logging.error(f"No image files found in folder '{INPUT_FOLDER}'.")
        return

    logging.info(f"Found {len(image_files)} image files to process.")
    
    processed_count = 0
    error_count = 0
    datas = [key, endpoint, type_dict[type_doc]]
    
    for image_path in image_files:
        try:
            process_image(image_path, prediction_url, headers, font, datas)
            processed_count += 1
            logging.info(f"Progress: {processed_count}/{len(image_files)} images processed")
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            error_count += 1
    
    logging.info("\n--- Processing Summary ---")
    logging.info(f"Total images: {len(image_files)}")
    logging.info(f"Successfully processed: {processed_count}")
    logging.info(f"Errors encountered: {error_count}")

if __name__ == "__main__":
    main()
