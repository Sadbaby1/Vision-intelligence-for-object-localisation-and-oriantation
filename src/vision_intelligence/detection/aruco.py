import cv2
import cv2.aruco as aruco


class ArucoCalibrator:
    """
    Detects a fixed ArUco marker in the camera frame and uses its known
    real-world size to calculate a pixel-to-millimeter conversion ratio.
    This ratio lets us later convert any object's pixel position into
    real-world millimeters for the robot.
    """

    def __init__(self, marker_length_mm=100, aruco_dict_type=aruco.DICT_5X5_100):
        # Real-world size of one side of the marker, in millimeters 
        self.marker_length_mm = marker_length_mm

        # Load the ArUco dictionary (must match the printed marker type, e.g. 5x5)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)

        # Default detector parameters (thresholding, refinement)
        self.parameters = aruco.DetectorParameters()

        # Create the detector object using the dictionary and parameters
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

        # Will hold the pixel-to-mm ratio once calculated (None until first successful calibration)
        self.pixel_mm_ratio = None

    def find_markers(self, frame):
        """
        Detects ArUco markers in the given frame.
        Returns the corner points and IDs of all detected markers.
        Also draws the detected marker outline directly on the frame.
        """
        # Convert to grayscale - marker detection works on intensity, not color
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers: corners = 4 corner points per marker, ids = marker ID numbers
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None:
            # Draw a box + ID label around each detected marker (visual feedback)
            aruco.drawDetectedMarkers(frame, corners, ids)
            print(f"Detected ArUco markers: {ids.flatten()}")

        return corners, ids

    def calculate_ratio(self, corners):
        """
        Calculates the pixel-to-millimeter ratio using the first detected marker.
        Formula: (perimeter in pixels) / (real perimeter in mm)
        Since perimeter = 4 sides, real perimeter = marker_length_mm * 4.
        """
        if not corners:
            # No marker detected in this frame - can't calculate anything
            return None

        # Measure the marker's perimeter in pixels (corners[0] = first detected marker)
        perimeter = cv2.arcLength(corners[0], True)

        # Convert pixel perimeter to a pixel/mm ratio using the known real-world size
        self.pixel_mm_ratio = perimeter / (self.marker_length_mm * 4)

        return self.pixel_mm_ratio
    
    def get_marker_center(self, corners):
        """
        Calculates the pixel center of the first detected marker,
        using the average of its 4 corner points.
        """
        if not corners:
            return None

        marker_corners = corners[0][0]  # shape: 4 points, each (x, y)
        center_x = marker_corners[:, 0].mean()
        center_y = marker_corners[:, 1].mean()

        return (center_x, center_y)
