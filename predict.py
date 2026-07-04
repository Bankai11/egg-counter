import os
import cv2
import numpy as np
from ultralytics import YOLO

def load_model(weights_path):
    """
    Load YOLOv8 model from weights path.
    If weights_path is a YOLO object, it returns it directly.
    """
    if isinstance(weights_path, YOLO):
        return weights_path
    if not os.path.exists(weights_path):
        # Fall back to yolov8n.pt if best.pt is not found yet
        print(f"Weights path {weights_path} not found. Loading pretrained yolov8n.pt...")
        return YOLO("yolov8n.pt")
    return YOLO(weights_path)

def predict_trays(image_source, model_or_path, conf_threshold=0.25, iou_threshold=0.45):
    """
    Performs egg tray detection on an image.
    
    Args:
        image_source (str or np.ndarray): Path to the image or loaded numpy image array.
        model_or_path (str or YOLO): Load weights from path or use preloaded YOLO object.
        conf_threshold (float): Bounding box confidence threshold.
        iou_threshold (float): Intersection Over Union threshold for NMS.
        
    Returns:
        tuple: (count, boxes, annotated_image)
            - count (int): number of detected egg trays.
            - boxes (list): list of tuples [(x1, y1, x2, y2, conf), ...]
            - annotated_image (np.ndarray): OpenCV image with visual overlays.
    """
    # Load model
    model = load_model(model_or_path)
    
    # Read image if path provided
    if isinstance(image_source, str):
        image = cv2.imread(image_source)
        if image is None:
            raise FileNotFoundError(f"Could not read image from {image_source}")
    else:
        image = image_source.copy()
        
    h, w, _ = image.shape
    
    # Run YOLO prediction
    # verbose=False reduces terminal noise
    results = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    # Extract boxes and count
    detected_boxes = []
    annotated_img = image.copy()
    
    result = results[0]
    
    # Check if there are detections
    if result.boxes is not None:
        for box in result.boxes:
            # Get coordinates, class, and confidence
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            # Ensure it belongs to our target class (egg_tray is class 0)
            if cls == 0:
                detected_boxes.append((x1, y1, x2, y2, conf))
                
                # Draw a premium glassmorphic/vibrant overlay
                # Vibrant neon cyan (BGR: 255, 191, 0)
                box_color = (255, 191, 0) # Cyan
                
                # Draw bounding box
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 3)
                
                # Draw bounding box corner accents for premium look
                length = min(30, int((x2-x1)*0.2), int((y2-y1)*0.2))
                # Top-Left corner
                cv2.line(annotated_img, (x1, y1), (x1 + length, y1), (0, 255, 255), 5)
                cv2.line(annotated_img, (x1, y1), (x1, y1 + length), (0, 255, 255), 5)
                # Top-Right corner
                cv2.line(annotated_img, (x2, y1), (x2 - length, y1), (0, 255, 255), 5)
                cv2.line(annotated_img, (x2, y1), (x2, y1 + length), (0, 255, 255), 5)
                # Bottom-Left corner
                cv2.line(annotated_img, (x1, y2), (x1 + length, y2), (0, 255, 255), 5)
                cv2.line(annotated_img, (x1, y2), (x1, y2 - length), (0, 255, 255), 5)
                # Bottom-Right corner
                cv2.line(annotated_img, (x2, y2), (x2 - length, y2), (0, 255, 255), 5)
                cv2.line(annotated_img, (x2, y2), (x2, y2 - length), (0, 255, 255), 5)

                # Draw label background
                label = f"egg_tray {conf:.2f}"
                (lbl_w, lbl_h), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                cv2.rectangle(annotated_img, (x1, y1 - lbl_h - 10), (x1 + lbl_w + 10, y1), box_color, -1)
                # Draw label text
                cv2.putText(annotated_img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
    tray_count = len(detected_boxes)
    
    # Overlay a premium count banner on top-left of the image
    banner_w = 260
    banner_h = 60
    
    # semi-transparent dark banner background
    overlay = annotated_img.copy()
    cv2.rectangle(overlay, (15, 15), (15 + banner_w, 15 + banner_h), (20, 20, 20), -1)
    # Apply alpha transparency blending
    cv2.addWeighted(overlay, 0.75, annotated_img, 0.25, 0, annotated_img)
    
    # Draw border on the banner
    cv2.rectangle(annotated_img, (15, 15), (15 + banner_w, 15 + banner_h), (0, 255, 255), 2)
    
    # Put Text: "Trays Detected: N"
    cv2.putText(
        annotated_img,
        f"Trays Detected: {tray_count}",
        (30, 52),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )
    
    return tray_count, detected_boxes, annotated_img

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test YOLOv8 egg tray detection on an image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to weights file (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--output", type=str, default="output_annotated.jpg", help="Path to save annotated output image")
    
    args = parser.parse_args()
    
    try:
        count, boxes, annotated_img = predict_trays(
            args.image,
            args.weights,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        
        print(f"Detected trays: {count}")
        print(f"Boxes: {boxes}")
        
        cv2.imwrite(args.output, annotated_img)
        print(f"Saved annotated image to: {args.output}")
        
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()
