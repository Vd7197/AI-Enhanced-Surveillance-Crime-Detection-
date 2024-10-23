!pip install ultralytics
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from google.colab import files
from IPython.display import display, Video

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')

# Define dangerous objects for labeling (expanded for more crime-related items)
danger_objects = ['knife', 'gun', 'rifle', 'bat', 'explosive', 'sword', 'grenade', 'hammer']

# Upload the video file to Colab
uploaded = files.upload()
video_path = next(iter(uploaded))

# Initialize video capture using the uploaded video
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25

# Define the codec and create VideoWriter object to save the output video
output_path = 'annotated_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Get the names of the detected objects from the model
names = model.names

# Define a function to check for suspicious activity based on detected objects
def is_suspicious_activity(boxes, names):
    for box in boxes:
        class_id = int(box.cls.item())
        if names[class_id] in danger_objects:
            return True
    return False

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model.predict(frame, conf=0.5, verbose=False)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        bboxes = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        # Iterate over detected boxes
        for bbox, confidence, class_id in zip(bboxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, bbox)
            label = names[class_id]

            # Handle misclassification if necessary (enhanced handling)
            if label == 'snowboard' and confidence < 0.6:
                label = 'knife'
            elif label == 'spatula' and confidence < 0.6:
                label = 'knife'
            elif label == 'hair drier' and confidence < 0.8:
                label = 'gun'

            # Draw bounding box and label, using red for dangerous objects
            color = (0, 0, 255) if label in danger_objects else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Save labeled images for analysis (e.g., save images with guns)
            if label == 'gun':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                gun_image_path = f"gun_{timestamp}.jpg"
                cv2.imwrite(gun_image_path, frame[y1:y2, x1:x2])

        # Check for suspicious activity
        if is_suspicious_activity(boxes, names):
            # Display warning message on the video frame
            cv2.putText(frame, "ALARM: Suspicious Activity Detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            # Print the alarm message (this could be used for triggering external alarms)
            print("ALARM: Suspicious activity detected! Possible weapon or dangerous object.")

    # Write the annotated frame to the output video
    out.write(frame)

# Release video resources
cap.release()
out.release()

# Download the video file
files.download(output_path)

# Display the annotated video
display(Video(output_path, embed=True))

# Confirm video saving process
print("Annotated video saved as:", output_path)
