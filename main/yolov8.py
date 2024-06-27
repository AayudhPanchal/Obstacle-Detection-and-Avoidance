import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model
model = YOLO('../yolov8n.pt')  # You can use yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt based on your need

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define different colors for different classes
class_colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Capture video from the webcam 
# To use webcam
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

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
        label = model.names[cls]

        detections.append({
            'label': f"{label} {conf:.2f}",
            'bbox': [x1, y1, x2 - x1, y2 - y1],
            'color': class_colors[cls]
        })

    for detection in detections:
        x, y, w, h = detection['bbox']
        label = detection['label']
        color = detection['color']

        # Draw bounding box with rounded corners
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

        # Draw label text with background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = x
        text_y = y - 10 if y - 10 > 10 else y + 10
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 4), (text_x + text_size[0], text_y), color, -1)
        cv2.putText(frame, label, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Resize the frame for display
    frame = cv2.resize(frame, (width * 2, height * 2))  # Double the size of the frame

    cv2.imshow("YOLOv8 Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()