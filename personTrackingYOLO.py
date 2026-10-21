from  ultralytics import YOLO
import cv2
model=YOLO("yolov8n.pt")
results=model.track(source=0,stream=True,conf=0.5,classes=[0],tracker="bytetrack.yaml")
seeds_id=set()
for result in results:
    if result.boxes.id is not None:
        ids=result.boxes.id.cpu().numpy().astype(int)
        for person_id in ids:
            seeds_id.add(person_id)
        print(f"Currently visible:{len(ids)}|Total Unique persons seen:{len(seeds_id)}")