import cv2
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            # Look for "person" (class 0) or "dog" (class 16)
            # You can check model.names to find other ID numbers
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            if class_id == 0 and conf > 0.5:
                cv2.imwrite("../detection_alert.jpg", frame)
                print("Person detected! Image saved.")

    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Smart Guard", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
