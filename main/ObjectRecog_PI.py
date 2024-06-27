import cv2

# Thresholds to detect object
confidence_threshold = 0.4  # Higher confidence threshold for better accuracy
nms_threshold = 0.3        # Non-maximum suppression threshold

# Load class names
classNames = []
classFile = "coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Load the neural network model
configPath = "cfg/ssd_mobilenet_v3_large_coco.pbtxt"
weightsPath = "weights/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, confidence_threshold, nms_threshold, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=confidence_threshold, nmsThreshold=nms_threshold)
    if len(objects) == 0: 
        objects = classNames
         
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if 0 < classId <= len(classNames):
                className = classNames[classId - 1]
                if className in objects:
                    objectInfo.append([box, className])
                    if draw:
                        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                        cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                
    return img, objectInfo

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break
            
        result, objectInfo = getObjects(img, confidence_threshold, nms_threshold)
        cv2.imshow("Output", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
