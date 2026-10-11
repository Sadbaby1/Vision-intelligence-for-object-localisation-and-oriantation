from ultralytics import YOLO
import cv2 as cv
from .detected_object import DetectedObject
from ..orientation.orientation import OrientationEstimator


class Detector:
    """
    Loads a YOLO model and runs inference on frames to detect
    objects, pixel position, and orientation of target objects.
    """

    def __init__(self, model_path="yolov11n.pt"):

        self.model = YOLO(model_path)
        

    def detect(self, frame):
        results = self.model.predict(frame)
        detected_objects = []

        # Loop through the results and extract relevant information
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = box.conf[0].item()

                # Only consider detections with confidence above a threshold (e.g., 0.5)
                if confidence > 0.5:
                    label = self.model.names[class_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Create a DetectedObject instance for each detected object
                    obj = DetectedObject(label, (x1, y1, x2, y2), confidence)
                    obj.orientation = self.orientation_estimator.estimate(
                        frame, (x1, y1, x2, y2)
                    )
                    detected_objects.append(obj)

                    # Draw bounding box and label on the frame
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_conf = f"{label} {confidence:.2f}"
                    cv.putText(
                        frame,
                        label_conf,
                        (x1, y1 - 10),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

        return detected_objects