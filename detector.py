# detector.py

from ultralytics import YOLO

class StickerDetector:

    def __init__(self, model_path, conf_threshold):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        return self.model(
            frame,
            conf=self.conf_threshold,
            iou=0.45,
            verbose=False
        )