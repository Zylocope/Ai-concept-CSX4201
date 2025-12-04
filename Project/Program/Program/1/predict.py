from ultralytics import YOLO

# Load a model
model = YOLO("last.pt")  # pretrained YOLO11n model

# List of image paths (update with your actual image paths)
image_paths = [
    r"labelme_output\YOLODataset\dataset_yolov10\test\images\1732814968214.jpg",
    r"labelme_output\YOLODataset\images\test\IMG_20241129_032437_836.jpg",
    r"labelme_output\YOLODataset\images\test\IMG_20241129_032438_422.jpg",
    r"labelme_output\YOLODataset\images\test\IMG_20241129_032620_688.png",
    r"labelme_output\YOLODataset\images\test\Messenger_creation_48CD7C52-E723-4C3F-820C-73490EA8F751.jpeg",
    r"labelme_output\YOLODataset\images\test\Screenshot 2024-11-29 140729.png",
    r"labelme_output\YOLODataset\images\test\Screenshot 2024-11-29 145956.png",
    r"labelme_output\YOLODataset\images\test\Screenshot_2024-11-29-01-07-40-924_com.google.android.gm.jpg"
]

# Run batched inference on the list of images
results = model(image_paths)  # return a list of Results objects

# Process results list
for i, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    
    result.show()  # display to screen
    result.save(filename=f"result_{i+1}.jpg")  # save to disk with unique filenames
