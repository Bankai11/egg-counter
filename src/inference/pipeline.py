import os
import cv2
import numpy as np
from ultralytics import YOLO
import sys

# Add src to path so we can import counting
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.counting.counter import StackCounter

class InferencePipeline:
    def __init__(self, detector_weights: str, conf_threshold: float = 0.45):
        """
        Initializes the Hybrid Egg Tray Counter Pipeline.
        """
        # Load Stage 1: Detector
        print(f"Loading YOLO detector from {detector_weights}...")
        self.detector = YOLO(detector_weights)
        self.conf_threshold = conf_threshold
        
        # Load Stage 2: Counter
        self.counter = StackCounter(prominence=3, min_distance_ratio=0.03)
        
    def process_image(self, image_path: str, output_path: str = None) -> dict:
        """
        Processes an image through the hybrid pipeline.
        Returns a dictionary with metrics and counts.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Stage 1: Detect stacks
        results = self.detector(img, conf=self.conf_threshold, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        
        total_trays = 0
        stack_results = []
        
        # Create output image
        out_img = img.copy()
        
        # Stage 2: Count trays in each detected stack
        for i, box in enumerate(boxes):
            # 1. Extract crop
            crop = self.counter.extract_crop(img, box)
            
            # 2. Apply 1D Signal Processing to count trays
            trays_in_stack, _, peaks = self.counter.count_trays_in_crop(crop)
            
            total_trays += trays_in_stack
            stack_results.append({
                "stack_id": i,
                "box": [float(b) for b in box],
                "trays": trays_in_stack
            })
            
            # 3. Visualize
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw text with background for readability
            text = f"Stack {i}: {trays_in_stack} trays"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out_img, (x1, y1 - 25), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(out_img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Optional: Draw lines for each peak (tray) inside the box
            for p in peaks:
                peak_y = int(y1 + p)
                cv2.line(out_img, (x1, peak_y), (x1 + 10, peak_y), (0, 0, 255), 2)
                
        # Draw total
        cv2.putText(out_img, f"TOTAL TRAYS: {total_trays}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    
        if output_path:
            cv2.imwrite(output_path, out_img)
            
        return {
            "total_trays": total_trays,
            "stacks_detected": len(boxes),
            "stack_details": stack_results
        }

if __name__ == "__main__":
    # Test on a sample image
    # We will use the base yolov8n.pt since training was aborted for time constraints
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    weights = os.path.join(base_dir, 'yolov8n.pt')
    
    # Grab a test image
    test_img = os.path.join(base_dir, 'dataset', 'test', 'images', 'tray_0065.jpg')
    out_img = os.path.join(base_dir, 'exports', 'pipeline_output.jpg')
    
    if os.path.exists(test_img):
        pipeline = InferencePipeline(weights)
        print("Running end-to-end pipeline...")
        result = pipeline.process_image(test_img, out_img)
        print(f"Results: {result['stacks_detected']} stacks, {result['total_trays']} total trays.")
        print(f"Saved visualization to {out_img}")
    else:
        print(f"Test image not found: {test_img}")
