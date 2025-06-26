import cv2
import numpy as np
from ultralytics import YOLO
import os


class ObjectDetection:
    def __init__(self, yolo_model_path="best.pt"):
        """Initialize object detection processor."""
        try:
            self.yolo_model = YOLO(yolo_model_path)
            self.class_names = self.yolo_model.names  # Get class names
        except Exception as e:
            raise RuntimeError(f"Error loading YOLO model: {e}")

        self.cap = cv2.VideoCapture(0)  # Open webcam
        if not self.cap.isOpened():
            raise FileNotFoundError("Error: Could not access the webcam.")

    def process_video_frame(self, frame):
        """Process each video frame to detect specific objects."""
        results = self.yolo_model(frame)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, class_id in zip(boxes, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = self.class_names[class_id] if class_id < len(self.class_names) else "Unknown"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return frame

    def start_processing(self):
        """Start real-time object detection using webcam."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = self.process_video_frame(frame)
            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = ObjectDetection()
    detector.start_processing()
