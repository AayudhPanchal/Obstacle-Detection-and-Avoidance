import cv2
import numpy as np

# Load YOLOv3
net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
layer_names = net.getLayerNames()

# Ensure compatibility with different versions of OpenCV
try:
    unconnected_layers = net.getUnconnectedOutLayers()
    if isinstance(unconnected_layers[0], list) or isinstance(unconnected_layers[0], np.ndarray):
        output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]
    else:
        output_layers = [layer_names[i - 1] for i in unconnected_layers]
except Exception as e:
    print(f"An error occurred: {e}")
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define different colors for different classes
class_colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize variables for distance calculation
    class_ids = []
    confidences = []
    boxes = []

    # Analyze the output
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"  # Include confidence in the label
            color = class_colors[class_ids[i]]

            # Simple distance estimation based on object size in the frame
            distance = (width * 20) / (w + 10)  # Adjust this formula based on your observations

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label} : {distance:.2f} inches', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Resize the frame for display
    frame = cv2.resize(frame, (width * 2, height * 2))  # Double the size of the frame

    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()