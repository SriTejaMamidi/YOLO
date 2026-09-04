from ultralytics import YOLO
import cv2
import pyttsx3
import time


# Load YOLO model
model = YOLO("yolov8n.pt")
# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
# Open webcam
cap = cv2.VideoCapture(0)
# To avoid repeating the same object continuously
last_spoken = ""
last_time = 0
speak_interval = 3  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Run YOLO detection
    results = model(frame)
    # Draw bounding boxes
    annotated_frame = results[0].plot()
    # Get detected objects
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        current_time = time.time()

        # Speak object name every 3 seconds
        if class_name != last_spoken or \
           (current_time - last_time) > speak_interval:

            print(f"Detected: {class_name}")

            engine.say(f"I can see a {class_name}")
            engine.runAndWait()

            last_spoken = class_name
            last_time = current_time

    cv2.imshow("YOLO Voice Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()