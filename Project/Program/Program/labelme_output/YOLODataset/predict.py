from ultralytics import YOLO

# Load a model
model = YOLO("YOLODataset\last.pt")  # pretrained YOLO11n model

# List of image paths (update with your actual image paths)
image_paths = [
    r"YOLODataset\images\test\IMG_20241129_032438_396.jpg",
    r"YOLODataset\images\test\IMG_20241129_032438_519.jpg",
    r"YOLODataset\images\test\IMG_20241129_032615_776.png",
    r"YOLODataset\images\test\Messenger_creation_738983EB-1169-4843-8443-D031C382A66F.jpeg",
    r"YOLODataset\images\test\Screenshot 2024-11-29 141844.png",
    r"YOLODataset\images\test\Screenshot 2025-03-01 200604.png",
    r"YOLODataset\images\test\Screenshot_2024-11-29-00-58-00-035_com.google.android.gm.jpg",
    r"YOLODataset\images\test\Screenshot 2024-11-29 091332.png"
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
