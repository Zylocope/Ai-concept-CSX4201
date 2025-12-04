import cv2
import os

def draw_bboxes(image, bboxes, class_ids):
    """
    Draw bounding boxes on an image. Bboxes should be in YOLO format (x_center, y_center, width, height).
    Args:
        image (numpy.ndarray): The image on which to draw, in RGB format.
        bboxes (list of tuples): List of bounding boxes in YOLO format.
        class_ids (list of ints): List of class IDs for the bounding boxes.
    Returns:
        numpy.ndarray: The image with bounding boxes drawn on it.
    """
    for bbox, class_id in zip(bboxes, class_ids):
        x_center, y_center, width, height = bbox
        x_center *= image.shape[1]
        y_center *= image.shape[0]
        width *= image.shape[1]
        height *= image.shape[0]

        top_left = int(x_center - width / 2), int(y_center - height / 2)
        bottom_right = int(x_center + width / 2), int(y_center + height / 2)

        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Green box with 2px thickness
        cv2.putText(image, str(class_id), (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

def visualize_augmented_images(output_folder):
    """
    Read augmented images and their bounding box data from a specified folder, then draw the bounding boxes.
    Args:
        output_folder (str): The folder path containing augmented images and .txt annotation files.
    """
    for file in os.listdir(output_folder):
        if file.endswith('.jpg'):
            image_path = os.path.join(output_folder, file)
            txt_path = os.path.join(output_folder, file.replace('.jpg', '.txt'))

            if not os.path.exists(txt_path):
                continue  # Skip if corresponding TXT file does not exist

            # Load image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

            # Load bounding boxes
            bboxes = []
            class_ids = []
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = parts[0]
                    class_id = float(class_id)
                    class_id = int(class_id)
                    box = tuple(map(float, parts[1:]))
                    bboxes.append(box)
                    class_ids.append(class_id)

            # Draw bounding boxes on the image
            visualized_image = draw_bboxes(image.copy(), bboxes, class_ids)

            # Show the image
            cv2.imshow('Augmented Image with Bounding Boxes', cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)  # Wait for a key press to move to the next image
            cv2.destroyAllWindows()

# Specify the output folder path
output_folder = 'output'
visualize_augmented_images(output_folder)
