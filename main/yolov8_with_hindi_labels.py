import cv2
from ultralytics import YOLO
import numpy as np
from translate import Translator
from PIL import ImageFont, ImageDraw, Image
import os
import time

# Load YOLOv8 model
model = YOLO('models/yolov8n.pt')

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define different colors for different classes
class_colors = np.random.uniform(0, 255, size=(len(classes), 3)).astype(int)

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Initialize the translator with caching
translator = Translator(to_lang="hi")
translation_cache = {}

# Load the Unicode font
font_path = "fonts/NotoSansDevanagari.ttf"
if not os.path.isfile(font_path):
    raise FileNotFoundError(f"The font file '{font_path}' was not found. Please ensure the path is correct.")
font = ImageFont.truetype(font_path, 20)

def calculate_distance(w):
    distance = (width * 20) / (w + 10)  # Adjust this formula based on your observations
    return distance

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Resize the frame to a smaller size for faster processing
    small_frame = cv2.resize(frame, (320, 240))

    # Perform detection
    results = model(small_frame)[0]
    detections = []

    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
        conf = r.conf[0].item()
        cls = int(r.cls[0].item())
        label = model.names[cls]

        # Translate the label to Hindi with caching
        if label not in translation_cache:
            translation_cache[label] = translator.translate(label)
        translated_label = translation_cache[label]

        # Scale bounding boxes back to original frame size
        x1 = int(x1 * (width / 320))
        y1 = int(y1 * (height / 240))
        x2 = int(x2 * (width / 320))
        y2 = int(y2 * (height / 240))

        detections.append({
            'label': f"{translated_label} {conf:.2f}",
            'bbox': [x1, y1, x2 - x1, y2 - y1],
            'color': class_colors[cls]
        })

    for detection in detections:
        x, y, w, h = detection['bbox']
        label = detection['label']
        color = tuple(detection['color'].tolist())
        distance = calculate_distance(w)

        display_text = f"{label} : {distance:.2f}"

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

        # Convert the frame to a PIL image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Draw label text with background using Pillow
        text_bbox = draw.textbbox((0, 0), display_text, font=font)
        text_x = x
        text_y = y - 10 if y - 10 > 10 else y + 10
        draw.rectangle([text_x, text_y - text_bbox[3], text_x + text_bbox[2], text_y], fill=color)
        draw.text((text_x, text_y - text_bbox[3]), display_text, font=font, fill=(255, 255, 255))

        # Convert the PIL image back to an OpenCV image
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Calculate and display FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"CISMR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("YOLOv8 Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
