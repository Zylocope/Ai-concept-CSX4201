import cv2
import albumentations as A
import os

# Define the Albumentations transform with a broader range of adjustments
transform = A.Compose(
    [
        # Color and lighting adjustment
        A.RandomBrightnessContrast(p=0.5),  # Adjust brightness and contrast
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),  # Color variation
        
        # Geometric transforms
        A.HorizontalFlip(p=0.5),  # Horizontal flip
        A.VerticalFlip(p=0.5),    # Vertical flip
        A.Affine(
            rotate=(-45, 45),  # Rotation
            translate_percent=(0.1, 0.1),  # Translation percentage
            scale=(0.9, 1.1),  # Scaling
            shear=(-16, 16),   # Shearing
            p=0.5
        ),  # General affine transformations

        # Cropping and resizing
        A.RandomCrop(height=200, width=400, p=0.5),  # Random crop
        A.Resize(height=450, width=450, p=1),  # Resize all images to 450x450

        # # Advanced effects
        # A.GaussianBlur(blur_limit=3, p=0.5),  # Apply Gaussian Blur
        # A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # Add Gaussian noise
        # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),  # Elastic transformation for non-rigid transformation

        # # Weather effects
        # A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, alpha_coef=0.1, p=0.5),  # Simulate fog
        # A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, p=0.5),  # Simulate rain
        # A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, p=0.5),  # Simulate snow
        # A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, p=0.5),  # Sun flare

        # Normalization
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1)  # Normalize for pre-trained models
    ],
    bbox_params=A.BboxParams(
        format='yolo',  # Specify that we're using YOLO format for bounding boxes
        label_fields=['class_ids']  # Link class IDs to the bboxes
    )
)

# Function to load and process images and bounding boxes
def process_images():
    for entry in os.listdir('images'):
        if entry.endswith('.jpg'):
            filename = entry[:-4]
            image_path = f'images/{entry}'
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

            txt_path = f'images/{filename}.txt'
            bboxes = []
            class_ids = []
            with open(txt_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    box = tuple(map(float, parts[1:]))
                    bboxes.append(box)
                    class_ids.append(class_id)

            # Apply transformations
            transformed = transform(image=image, bboxes=bboxes, class_ids=class_ids)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_ids = transformed['class_ids']

            # Save results
            if not os.path.exists('output'):
                os.makedirs('output')
            cv2.imwrite(f'output/{filename}_aug.jpg', cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))  # Save augmented image
            with open(f'output/{filename}_aug.txt', 'w') as f:  # Save augmented boxes
                for class_id, bbox in zip(transformed_class_ids, transformed_bboxes):
                    f.write(f"{class_id} {' '.join(map(str, bbox))}\n")

# Execute the function
process_images()
