import os
import shutil
 
def reorganize_dataset(base_path):
    # Define the original and new paths
    orig_images_path = os.path.join(base_path, "images")
    orig_labels_path = os.path.join(base_path, "labels")
    new_base_path = os.path.join(base_path, "dataset_yolov10")
   
    # Create new directories if they don't exist
    os.makedirs(new_base_path, exist_ok=True)
 
    # Copy files from original to new structure and rename folders
    for dtype in ["train", "val", "test"]:
        new_images_path = os.path.join(new_base_path, dtype, "images")
        new_labels_path = os.path.join(new_base_path, dtype, "labels")
 
        # Make directories
        os.makedirs(new_images_path, exist_ok=True)
        os.makedirs(new_labels_path, exist_ok=True)
 
        # Copy and rename image files
        src_images = os.path.join(orig_images_path, dtype)
        src_labels = os.path.join(orig_labels_path, dtype)
       
        for filename in os.listdir(src_images):
            shutil.copy(os.path.join(src_images, filename), new_images_path)
       
        # Copy and rename label files (if using YOLO, might be txt files instead of JSON)
        if os.path.exists(src_labels):
            for filename in os.listdir(src_labels):
                shutil.copy(os.path.join(src_labels, filename), new_labels_path)
 
    print(f"Reorganized dataset available at: {new_base_path}")
 
# Replace '/path/to/dataset/YOLODataset' with the actual path to your 'YOLODataset' folder
reorganize_dataset("C:/Users/lamin/OneDrive/Desktop/4201_AI/INCLASS/dataset/YOLODataset")
