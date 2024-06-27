import cv2
from ultralytics import YOLO
import numpy as np
import time
from collections import deque

# Load YOLOv8 model
model = YOLO('models/yolov8n.pt')  # Choose the appropriate YOLOv8 model based on your requirements

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define different colors for different classes
class_colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Initialize variables for object tracking
object_tracker = {}  # Dictionary to store object tracking information
next_object_id = 0  # Counter to assign unique IDs to detected objects

def calculate_distance(w, width):
    distance = (width * 20) / (w + 10)  # Adjust this formula based on your observations
    return distance

def calculate_speed(tracker, object_id, new_position, current_time, pixel_to_meter_ratio, min_movement=2, min_time_interval=0.1):
    if object_id in tracker and len(tracker[object_id]) > 1:
        old_position, old_time = tracker[object_id][-1]  # Get the last known position and time
        distance_moved_pixels = np.linalg.norm(np.array(new_position) - np.array(old_position))
        if distance_moved_pixels < min_movement:
            return 0  # Ignore very small movements
        distance_moved_meters = distance_moved_pixels * pixel_to_meter_ratio  # Convert pixels to meters
        time_elapsed = current_time - old_time
        if time_elapsed < min_time_interval:
            return 0  # Ignore very short time intervals
        if time_elapsed > 0:
            speed = distance_moved_meters / time_elapsed  # Speed in meters per second
            return speed
    return 0

# Define the conversion ratio (e.g., 0.01 meters per pixel)
pixel_to_meter_ratio = 0.01

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Perform object detection
    results = model(frame)

    # List to store current detections
    detections = []

    for result in results:
        for r in result.boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            conf = r.conf.item()
            cls = int(r.cls.item())

            # Filter out detections with low confidence
            if conf < 0.4:  # Adjust the threshold as needed
                continue

            label = f"{classes[cls]} {conf:.2f}"

            # Calculate bounding box width and height
            w = x2 - x1
            h = y2 - y1

            # Calculate the center of the bounding box
            center_x, center_y = x1 + w // 2, y1 + h // 2

            # Try to match this detection to an existing object
            matched_object_id = None
            min_distance = float('inf')

            for object_id, positions in object_tracker.items():
                if len(positions) > 0:
                    last_position, _ = positions[-1]
                    distance = np.linalg.norm(np.array((center_x, center_y)) - np.array(last_position))
                    if distance < min_distance:
                        min_distance = distance
                        matched_object_id = object_id

            # If no match found, assign a new object ID
            if matched_object_id is None or min_distance > 50:  # Distance threshold for matching
                matched_object_id = next_object_id
                next_object_id += 1

            # Calculate distance and speed
            distance = calculate_distance(w, width)
            current_time = time.time()
            speed = calculate_speed(object_tracker, matched_object_id, (center_x, center_y), current_time, pixel_to_meter_ratio)

            display_label = f"{label} : {distance:.2f}cm : {speed:.2f} m/s"

            # Update object tracker with current position and time
            if matched_object_id not in object_tracker:
                object_tracker[matched_object_id] = deque(maxlen=2)
            object_tracker[matched_object_id].append(((center_x, center_y), current_time))

            # Draw bounding box with rounded corners
            cv2.rectangle(frame, (x1, y1), (x2, y2), class_colors[cls], 2, cv2.LINE_AA)

            # Draw label text with background
            text_size, _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.2, 1)
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.rectangle(frame, (text_x, text_y - text_size[1] - 4), (text_x + text_size[0], text_y), class_colors[cls], -1)
            cv2.putText(frame, display_label, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Resize the frame for display
    frame = cv2.resize(frame, (width * 2, height * 2))  # Double the size of the frame

    cv2.imshow("YOLOv8 Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
