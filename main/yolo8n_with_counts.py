import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# Load YOLOv8 model
model = YOLO('models/yolov8n.pt')

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define different colors for different classes
class_colors = np.random.uniform(0, 255, size=(len(classes), 3)).astype(int)

# Capture video from the webcam
cap = cv2.VideoCapture(0)

def calculate_distance(w):
    distance = (width * 20) / (w + 10)  # Adjust this formula based on your observations
    return distance

# Initialize object tracker
tracker = defaultdict(lambda: [None, None])  # Dictionary to store the last position and ID of each object
object_id = 0  # Counter to assign unique IDs to detected objects
object_count = defaultdict(int)  # To keep track of the number of each type of object

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Perform detection
    results = model(frame)[0]
    detections = []

    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
        conf = r.conf[0].item()
        cls = int(r.cls[0].item())
        label = classes[cls]

        detections.append({
            'label': label,
            'bbox': [x1, y1, x2 - x1, y2 - y1],
            'color': tuple(class_colors[cls].tolist()),
            'confidence': conf
        })

    # Reset object count for each frame
    current_count = defaultdict(int)

    # Track objects and assign IDs
    current_objects = {}
    for detection in detections:
        x, y, w, h = detection['bbox']
        label = detection['label']
        found_match = False

        for obj_id, (prev_bbox, prev_label) in tracker.items():
            if prev_bbox is not None and prev_label == label:
                prev_x, prev_y, prev_w, prev_h = prev_bbox
                if abs(x - prev_x) < 50 and abs(y - prev_y) < 50:  # Adjust threshold for matching
                    current_objects[obj_id] = (detection['bbox'], detection['label'])
                    found_match = True
                    break

        if not found_match:
            current_objects[object_id] = (detection['bbox'], detection['label'])
            object_id += 1

    tracker = current_objects  # Update tracker with current frame's objects

    # Number objects by type
    numbered_labels = {}
    for obj_id, (bbox, label) in tracker.items():
        current_count[label] += 1
        numbered_labels[obj_id] = f"{label} {current_count[label]}"

    for obj_id, (bbox, label) in tracker.items():
        x, y, w, h = bbox
        color = tuple(class_colors[classes.index(label)].tolist())
        distance = calculate_distance(w)
        label_text = f"{numbered_labels[obj_id]}: {distance:.2f}cm"

        # Draw bounding box with rounded corners
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

        # Draw label text with background
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = x
        text_y = y - 10 if y - 10 > 10 else y + 10
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 4), (text_x + text_size[0], text_y), color, -1)
        cv2.putText(frame, label_text, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Resize the frame for display
    frame = cv2.resize(frame, (width * 2, height * 2))  # Double the size of the frame

    cv2.imshow("YOLOv8 Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
