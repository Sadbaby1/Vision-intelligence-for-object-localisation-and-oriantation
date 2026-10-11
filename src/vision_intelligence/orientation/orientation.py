import cv2
import numpy as np


class OrientationEstimator:
    """
    Estimates the rotation angle of a detected object using contour analysis.
    Crops the object's region from the frame, finds its outline, and fits
    a minimum-area rectangle to determine the angle.
    """

    def estimate(self, frame, bbox):
        x1, y1, x2, y2 = bbox

        # Crop the region containing the object
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Convert to grayscale and threshold to isolate the shape from background
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours in the thresholded region
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Assume the largest contour is the object itself
        largest_contour = max(contours, key=cv2.contourArea)

        # Fit the smallest rotated rectangle around the contour
        rect = cv2.minAreaRect(largest_contour)  # returns (center, (w, h), angle)
        angle = rect[2]

        return angle
