from ultralytics import YOLO

yolo = YOLO("yolov8n.pt", task="detect")

results = yolo(source="./data/images/bus.jpg")