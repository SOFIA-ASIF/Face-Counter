import cv2   #for computer vision tasks
from ultralytics import YOLO #for YOLOv8 object detection
from collections import defaultdict # to count objects

# Load YOLOv8 model
model = YOLO('yolov8s.pt')  # We can also use 'yolov8m.pt' or 'yolov8l.pt' for better accuracy

# Get class names from the model
class_names = model.names

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    results = model(frame, stream=True)
    object_counts = defaultdict(int)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            class_name = class_names[cls]
            object_counts[class_name] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display all object counts on the frame
    y_offset = 30
    for obj_name, count in object_counts.items():
        cv2.putText(frame, f'{obj_name}: {count}', (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        y_offset += 30

    cv2.imshow("Object Detection and Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
