from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")
app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World"}


@app.post("/detect/")
async def detect_objects(file: UploadFile):
    # Process the uploaded image for object detection
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Perform object detection with YOLOv8
    detections = model.predict(image)

    return {"detections": detections}