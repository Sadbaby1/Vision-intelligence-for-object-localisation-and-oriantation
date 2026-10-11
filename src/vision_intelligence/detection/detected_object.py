class DetectedObject:
    """
    Represents a single object detected by the AI model.
    Acts as a clean data container passed between detector,
    coordinate_transform, and server modules.
    """

    def __init__(self, label, bbox, confidence, orientation=None):
        self.label = label  # shape name (e.g. "filter", "frame"
        self.confidence = confidence  # detection confidence score
        self.orientation = orientation  # angle in degrees, set later
        self.world_x = None  # filled in by coordinate_transform.py
        self.world_y = None

    def center(self):
        """Returns the pixel center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def __str__(self):
        return (
            f"{self.label} (conf={self.confidence:.2f}) "
            f"at pixel {self.center()}, orientation={self.orientation}"
        )
