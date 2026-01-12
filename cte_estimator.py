"""CTE (Cross-Track Error) Estimator classes.

This module provides CTE estimation classes for detecting track boundaries
and centerlines from camera images.

Classes:
    CTEEstimator: Base class for CTE estimation without visualization
    VisualCTEEstimator: CTE estimator with visualization and debugging support
"""

from typing import Optional, Tuple

import cv2
import numpy as np


class CTEEstimator:
    """Base class for estimating cross-track error (CTE) from camera images.
    
    This class provides the core CTE estimation logic without visualization.
    """
    
    def __init__(
        self,
        method: str = "canny_edges",
        image_width: int = 320,
        image_height: int = 240,
        max_cte: float = 3.0,
        # Color thresholds for track detection (HSV) - used by both color_edge_detection and centerline_tracking
        track_lower: Tuple[int, int, int] = (0, 0, 200),
        track_upper: Tuple[int, int, int] = (180, 30, 255),
    ):
        """Initialize the CTE estimator.
        
        Args:
            method: Estimation method:
                - "canny_edges": Canny edge detection (default)
                - "color_edge_detection": HSV color-based edge detection
                - "centerline_tracking": HSV color-based centerline tracking (uses same HSV as color_edge_detection)
            image_width: Expected image width
            image_height: Expected image height
            max_cte: Maximum CTE value (for normalization)
            track_lower: HSV lower bound for track color (used by color_edge_detection and centerline_tracking)
            track_upper: HSV upper bound for track color (used by color_edge_detection and centerline_tracking)
        """
        self.method = method
        self.image_width = image_width
        self.image_height = image_height
        self.image_center = image_width // 2
        self.max_cte = max_cte
        
        self.track_lower = np.array(track_lower)
        self.track_upper = np.array(track_upper)
    
    def estimate(self, frame_bgr: np.ndarray) -> Tuple[float, float]:
        """Estimate CTE from camera image.
        
        Args:
            frame_bgr: Camera image in BGR format (HWC uint8)
        
        Returns:
            Tuple of (cte, confidence):
            - cte: Cross-track error (positive = right of center)
            - confidence: Detection confidence (0.0 to 1.0)
        """
        if self.method == "canny_edges":
            return self._estimate_by_canny(frame_bgr)
        elif self.method == "color_edge_detection":
            return self._estimate_by_color_edges(frame_bgr)
        elif self.method == "centerline_tracking":
            return self._estimate_by_centerline(frame_bgr)
        elif self.method == "edge_detection":
            # Backward compatibility: map to canny_edges
            return self._estimate_by_canny(frame_bgr)
        else:
            # Unknown method
            return 0.0, 0.0
    
    def _estimate_by_canny(self, frame_bgr: np.ndarray) -> Tuple[float, float]:
        """Estimate CTE using Canny edge detection.
        
        Detects left and right track boundaries using Canny edges,
        computes their midpoint, and returns the offset from image center as CTE.
        """
        h, w = frame_bgr.shape[:2]
        
        # Take lower portion of image (near the car)
        roi_start = int(h * 0.6)
        roi = frame_bgr[roi_start:, :]
        
        # Convert to grayscale and apply edge detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find edge points in bottom rows
        scan_row = edges.shape[0] - 10  # Near bottom
        if scan_row < 0:
            scan_row = edges.shape[0] // 2
        
        edge_pixels = np.where(edges[scan_row, :] > 0)[0]
        
        if len(edge_pixels) < 2:
            # Not enough edges detected
            return 0.0, 0.0
        
        # Assume leftmost and rightmost edges are track boundaries
        left_edge = edge_pixels[0]
        right_edge = edge_pixels[-1]
        
        # Compute lane center
        lane_center = (left_edge + right_edge) // 2
        
        # Compute CTE (normalized)
        pixel_offset = lane_center - (w // 2)
        cte = (pixel_offset / (w / 2)) * self.max_cte
        
        # Confidence based on edge separation
        edge_width = right_edge - left_edge
        expected_width = w * 0.5  # Expect track to be ~50% of image width
        confidence = min(1.0, edge_width / expected_width)
        
        return float(cte), float(confidence)
    
    def _estimate_by_color_edges(self, frame_bgr: np.ndarray) -> Tuple[float, float]:
        """Estimate CTE by detecting track edges using HSV color thresholds.
        
        Detects left and right track boundaries by color (e.g., white lines),
        computes their midpoint, and returns the offset from image center as CTE.
        """
        h, w = frame_bgr.shape[:2]
        
        # Take lower portion of image (near the car)
        roi_start = int(h * 0.6)
        roi = frame_bgr[roi_start:, :]
        
        # Convert to HSV and detect edge color
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.track_lower, self.track_upper)
        
        # Find edge points in bottom rows
        scan_row = max(0, mask.shape[0] - 10)
        edge_pixels = np.where(mask[scan_row, :] > 0)[0]
        
        if len(edge_pixels) < 2:
            # Not enough edges detected
            return 0.0, 0.0
        
        # Assume leftmost and rightmost edges are track boundaries
        left_edge = edge_pixels[0]
        right_edge = edge_pixels[-1]
        
        # Compute lane center
        lane_center = (left_edge + right_edge) // 2
        
        # Compute CTE (normalized)
        pixel_offset = lane_center - (w // 2)
        cte = (pixel_offset / (w / 2)) * self.max_cte
        
        # Confidence based on edge separation
        edge_width = right_edge - left_edge
        expected_width = w * 0.4  # Expect track to be ~40% of image width
        confidence = min(1.0, edge_width / expected_width)
        
        return float(cte), float(confidence)
    
    def _estimate_by_centerline(self, frame_bgr: np.ndarray) -> Tuple[float, float]:
        """Estimate CTE by tracking a colored centerline.
        
        Uses the same HSV thresholds as color_edge_detection, but computes
        the centroid of the entire detected color region instead of just edges.
        """
        h, w = frame_bgr.shape[:2]
        
        # Take lower portion of image
        roi_start = int(h * 0.5)
        roi = frame_bgr[roi_start:, :]
        
        # Convert to HSV and detect track color (same thresholds as color_edge_detection)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.track_lower, self.track_upper)
        
        # Find centroid of detected region
        moments = cv2.moments(mask)
        
        if moments["m00"] < 100:  # Not enough pixels detected
            return 0.0, 0.0
        
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        
        # Compute CTE
        pixel_offset = cx - (w // 2)
        cte = (pixel_offset / (w / 2)) * self.max_cte
        
        # Confidence based on detected area
        detected_area = moments["m00"]
        expected_area = w * (h - roi_start) * 0.05  # Expect ~5% of ROI
        confidence = min(1.0, detected_area / expected_area)
        
        return float(cte), float(confidence)


class VisualCTEEstimator(CTEEstimator):
    """CTE estimator with visualization and debugging support.
    
    Inherits from CTEEstimator and adds debug image and mask storage
    for visualization purposes.
    """
    
    def __init__(
        self,
        method: str = "canny_edges",
        image_width: int = 320,
        image_height: int = 240,
        max_cte: float = 3.0,
        # Color thresholds for track detection (HSV) - used by both color_edge_detection and centerline_tracking
        track_lower: Tuple[int, int, int] = (0, 0, 200),
        track_upper: Tuple[int, int, int] = (180, 30, 255),
    ):
        """Initialize the visual CTE estimator.
        
        Args:
            method: Estimation method:
                - "canny_edges": Canny edge detection (default)
                - "color_edge_detection": HSV color-based edge detection
                - "centerline_tracking": HSV color-based centerline tracking (uses same HSV as color_edge_detection)
            image_width: Expected image width
            image_height: Expected image height
            max_cte: Maximum CTE value (for normalization)
            track_lower: HSV lower bound for track color (used by color_edge_detection and centerline_tracking)
            track_upper: HSV upper bound for track color (used by color_edge_detection and centerline_tracking)
        """
        super().__init__(method, image_width, image_height, max_cte, track_lower, track_upper)
        
        # For visualization/debugging
        self.last_debug_image: Optional[np.ndarray] = None
        self.last_mask_image: Optional[np.ndarray] = None  # Mask used for CTE estimation
    
    def estimate(self, frame_bgr: np.ndarray) -> Tuple[float, float]:
        """Estimate CTE from camera image with visualization support.
        
        Args:
            frame_bgr: Camera image in BGR format (HWC uint8)
        
        Returns:
            Tuple of (cte, confidence):
            - cte: Cross-track error (positive = right of center)
            - confidence: Detection confidence (0.0 to 1.0)
        """
        if self.method == "canny_edges":
            return self._estimate_by_canny(frame_bgr)
        elif self.method == "color_edge_detection":
            return self._estimate_by_color_edges(frame_bgr)
        elif self.method == "centerline_tracking":
            return self._estimate_by_centerline(frame_bgr)
        elif self.method == "edge_detection":
            # Backward compatibility: map to canny_edges
            return self._estimate_by_canny(frame_bgr)
        else:
            # Unknown method, set mask to None
            self.last_mask_image = None
            return 0.0, 0.0
    
    def _estimate_by_canny(self, frame_bgr: np.ndarray) -> Tuple[float, float]:
        """Estimate CTE using Canny edge detection with visualization."""
        h, w = frame_bgr.shape[:2]
        
        # Take lower portion of image (near the car)
        roi_start = int(h * 0.6)
        roi = frame_bgr[roi_start:, :]
        
        # Convert to grayscale and apply edge detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Store edge detection mask for visualization
        self.last_mask_image = edges.copy()
        
        # Find edge points in bottom rows
        scan_row = edges.shape[0] - 10  # Near bottom
        if scan_row < 0:
            scan_row = edges.shape[0] // 2
        
        edge_pixels = np.where(edges[scan_row, :] > 0)[0]
        
        if len(edge_pixels) < 2:
            # Not enough edges detected
            self.last_debug_image = roi.copy()
            return 0.0, 0.0
        
        # Assume leftmost and rightmost edges are track boundaries
        left_edge = edge_pixels[0]
        right_edge = edge_pixels[-1]
        
        # Compute lane center
        lane_center = (left_edge + right_edge) // 2
        
        # Compute CTE (normalized)
        pixel_offset = lane_center - (w // 2)
        cte = (pixel_offset / (w / 2)) * self.max_cte
        
        # Confidence based on edge separation
        edge_width = right_edge - left_edge
        expected_width = w * 0.5  # Expect track to be ~50% of image width
        confidence = min(1.0, edge_width / expected_width)
        
        # Debug visualization
        self.last_debug_image = roi.copy()
        cv2.line(self.last_debug_image, (left_edge, scan_row), (left_edge, scan_row - 20), (0, 255, 0), 2)
        cv2.line(self.last_debug_image, (right_edge, scan_row), (right_edge, scan_row - 20), (0, 255, 0), 2)
        cv2.line(self.last_debug_image, (lane_center, scan_row), (lane_center, scan_row - 30), (0, 0, 255), 2)
        
        return float(cte), float(confidence)
    
    def _estimate_by_color_edges(self, frame_bgr: np.ndarray) -> Tuple[float, float]:
        """Estimate CTE by detecting track edges using HSV color thresholds with visualization."""
        h, w = frame_bgr.shape[:2]
        
        # Take lower portion of image (near the car)
        roi_start = int(h * 0.6)
        roi = frame_bgr[roi_start:, :]
        
        # Convert to HSV and detect edge color
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.track_lower, self.track_upper)
        
        # Store mask for visualization
        self.last_mask_image = mask.copy()
        
        # Find edge points in bottom rows
        scan_row = max(0, mask.shape[0] - 10)
        edge_pixels = np.where(mask[scan_row, :] > 0)[0]
        
        if len(edge_pixels) < 2:
            # Not enough edges detected
            self.last_debug_image = roi.copy()
            return 0.0, 0.0
        
        # Assume leftmost and rightmost edges are track boundaries
        left_edge = edge_pixels[0]
        right_edge = edge_pixels[-1]
        
        # Compute lane center
        lane_center = (left_edge + right_edge) // 2
        
        # Compute CTE (normalized)
        pixel_offset = lane_center - (w // 2)
        cte = (pixel_offset / (w / 2)) * self.max_cte
        
        # Confidence based on edge separation
        edge_width = right_edge - left_edge
        expected_width = w * 0.4  # Expect track to be ~40% of image width
        confidence = min(1.0, edge_width / expected_width)
        
        # Debug visualization
        self.last_debug_image = roi.copy()
        cv2.circle(self.last_debug_image, (left_edge, scan_row), 5, (0, 255, 0), -1)
        cv2.circle(self.last_debug_image, (right_edge, scan_row), 5, (0, 255, 0), -1)
        cv2.circle(self.last_debug_image, (lane_center, scan_row), 8, (0, 0, 255), -1)
        cv2.line(self.last_debug_image, (w // 2, 0), (w // 2, roi.shape[0]), (255, 255, 0), 2)
        
        return float(cte), float(confidence)
    
    def _estimate_by_centerline(self, frame_bgr: np.ndarray) -> Tuple[float, float]:
        """Estimate CTE by tracking a colored centerline with visualization.
        
        Uses the same HSV thresholds as color_edge_detection, but computes
        the centroid of the entire detected color region instead of just edges.
        """
        h, w = frame_bgr.shape[:2]
        
        # Take lower portion of image
        roi_start = int(h * 0.5)
        roi = frame_bgr[roi_start:, :]
        
        # Convert to HSV and detect track color (same thresholds as color_edge_detection)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.track_lower, self.track_upper)
        
        # Store mask for visualization
        self.last_mask_image = mask.copy()
        
        # Find centroid of detected region
        moments = cv2.moments(mask)
        
        if moments["m00"] < 100:  # Not enough pixels detected
            self.last_debug_image = roi.copy()
            return 0.0, 0.0
        
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        
        # Compute CTE
        pixel_offset = cx - (w // 2)
        cte = (pixel_offset / (w / 2)) * self.max_cte
        
        # Confidence based on detected area
        detected_area = moments["m00"]
        expected_area = w * (h - roi_start) * 0.05  # Expect ~5% of ROI
        confidence = min(1.0, detected_area / expected_area)
        
        # Debug visualization
        self.last_debug_image = roi.copy()
        cv2.circle(self.last_debug_image, (cx, cy), 10, (0, 0, 255), -1)
        cv2.line(self.last_debug_image, (w // 2, 0), (w // 2, roi.shape[0]), (255, 0, 0), 1)
        
        return float(cte), float(confidence)

