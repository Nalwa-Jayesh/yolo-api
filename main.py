from fastapi import FastAPI
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World"}