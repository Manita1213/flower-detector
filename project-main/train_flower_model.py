from ultralytics import YOLO

# مدل پایه (می‌تونی از yolov8s.pt یا yolov8m.pt هم استفاده کنی)
model = YOLO("yolov8n.pt")

# آموزش مدل
model.train(
    data="dataset/data.yaml",
    epochs=30,
    imgsz=640,
    project="flower_detector",
    name="yolov8_flower_model",
    val=True
)
