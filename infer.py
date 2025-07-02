from ultralytics import YOLO

model = YOLO("weights/best.pt")

results = model.predict(source="images_for_test/3_1_frame_98.jpg", save=True, conf=0.8)