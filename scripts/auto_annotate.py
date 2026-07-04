import os
import glob
import json
from ultralytics import YOLO
import sys

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_dir = os.path.join(root_dir, 'dataset_processed')
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    
    # Load pretrained YOLOv8n (COCO)
    model_path = os.path.join(root_dir, 'yolov8n.pt')
    if not os.path.exists(model_path):
        print(f"Error: Pretrained model not found at {model_path}")
        return
        
    print("Loading YOLOv8n model...")
    model = YOLO(model_path)
    
    review_queue = []
    total_images = 0
    total_annotations = 0
    
    # Find all images
    image_paths = glob.glob(os.path.join(images_dir, '**', '*.*'), recursive=True)
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_paths)} images to process.")
    
    for img_path in image_paths:
        total_images += 1
        
        # Determine corresponding label path
        rel_path = os.path.relpath(img_path, images_dir)
        # Change extension to .txt
        base, ext = os.path.splitext(rel_path)
        label_rel_path = base + '.txt'
        label_path = os.path.join(labels_dir, label_rel_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        
        # Run inference (low confidence to get *something*)
        results = model(img_path, conf=0.10, verbose=False)
        
        boxes = results[0].boxes
        
        if len(boxes) == 0:
            # No detections
            with open(label_path, 'w') as f:
                pass # Empty file
            review_queue.append({
                "image": rel_path,
                "reason": "0 detections"
            })
            continue
            
        max_conf = float(boxes.conf.max()) if len(boxes.conf) > 0 else 0.0
        
        # Write to label file
        with open(label_path, 'w') as f:
            for box in boxes:
                # box.xywhn is [x_center, y_center, width, height] normalized
                x, y, w, h = box.xywhn[0].tolist()
                # We map everything to class 0 ("stack") since yolov8n doesn't know what an egg tray is
                f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                total_annotations += 1
                
        if max_conf < 0.40:
            review_queue.append({
                "image": rel_path,
                "reason": f"Low max confidence ({max_conf:.2f} < 0.40)"
            })
            
    # Save review queue
    review_queue_path = os.path.join(dataset_dir, 'review_queue.json')
    with open(review_queue_path, 'w') as f:
        json.dump(review_queue, f, indent=4)
        
    # Print Report
    print("\n========================================")
    print("Auto-Annotation Report")
    print("========================================")
    print(f"Total Images Processed: {total_images}")
    print(f"Total Stack Annotations Generated: {total_annotations}")
    print(f"Average Stacks per Image: {total_annotations/max(1, total_images):.2f}")
    print(f"Images in Review Queue: {len(review_queue)}")
    print(f"Successfully Auto-Annotated (High Conf): {total_images - len(review_queue)}")
    print("========================================")
    print(f"Review queue saved to: {review_queue_path}")

    # Validate labels
    print("\nRunning Validation...")
    sys.path.append(os.path.dirname(__file__))
    from validate_labels import validate_labels
    valid = validate_labels(dataset_dir=dataset_dir, valid_classes={0})
    
    if not valid:
        sys.exit(1)

if __name__ == "__main__":
    main()
