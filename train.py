from ultralytics import YOLO

model = YOLO("yolo11x.pt")

model.train(
    data="dataset/data.yaml",
    epochs=30,
    imgsz=768,
    batch=1,
    name="food_detector",
    project="runs/train",
    workers=4,
    patience=20,
    val=True,
    translate=0.1,
    scale=0.5,
    flipud=0.5,
    fliplr=0.5,
    blur=0.1,
    augment=True
)