import cv2
import numpy as np
from scipy.signal import find_peaks

class StackCounter:
    def __init__(self, prominence=5, min_distance_ratio=0.05):
        """
        Args:
            prominence (int): Minimum prominence of a peak (ridge) to be counted as a tray.
            min_distance_ratio (float): Minimum distance between trays as a ratio of the crop height.
        """
        self.prominence = prominence
        self.min_distance_ratio = min_distance_ratio

    def extract_crop(self, image: np.ndarray, box: tuple) -> np.ndarray:
        """
        Extracts the stack crop given a bounding box (x1, y1, x2, y2).
        Handles out of bounds coordinates.
        """
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = image.shape[:2]
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        return image[y1:y2, x1:x2]

    def count_trays_in_crop(self, crop: np.ndarray) -> int:
        """
        Applies 1D signal processing to count horizontal ridges in a stack crop.
        """
        if crop.size == 0:
            return 0
            
        h, w = crop.shape[:2]
        
        # 1. Grayscale & Blur
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Canny Edge Detection (detect edges based on pilot study results)
        edge_map = cv2.Canny(blurred, 50, 150)
        
        # 3. Horizontal Projection Profile (Row sums)
        # Sum intensities along each row (axis=1)
        profile = np.sum(edge_map, axis=1)
        
        # Smooth the 1D signal to reduce noisy peaks
        # Using a moving average proportional to crop height
        window_size = max(3, int(h * 0.02))
        profile_smoothed = np.convolve(profile, np.ones(window_size)/window_size, mode='same')
        
        # 4. Peak Detection
        min_distance = max(1, int(h * self.min_distance_ratio))
        
        # Prominence dynamically scaled by the mean of the profile to handle different lighting
        dynamic_prominence = self.prominence * (np.mean(profile_smoothed) / 1000.0 + 1)
        
        peaks, _ = find_peaks(profile_smoothed, distance=min_distance, prominence=dynamic_prominence)
        
        return len(peaks), profile_smoothed, peaks
