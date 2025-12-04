

import cv2
from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Start capturing video from the camera (0 is the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    if not ret:
        print("Failed to grab frame")
        break

    # Perform object detection on the frame
    results = model(frame)

    # Display results on the frame
    annotated_frame = results[0].plot()  # Annotate the frame with detection results

    # Show the annotated frame in a window
    cv2.imshow("Object Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
