import cv2
from ultralytics import YOLO
model = YOLO("yolo11n-pose.pt")
cap = cv2.VideoCapture(0)
model.predict(source=0,show=True)
while True:
    ret, frame = cap.read()
    if not(ret): break
    result = model(frame)
    annotated_frame=result[0].plot()
    cv2.imshow('YOLO 2026 Pose Estimation',annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    cv2.destroyAllWindows()