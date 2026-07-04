import os
import glob
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_dir = os.path.join(root_dir, 'dataset_processed')
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    
    # We will use CUDA if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Grounding DINO on {device}...")
    
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    # Define our open-vocabulary prompt
    # Grounding DINO works best with lowercase text separated by periods
    text_prompt = "stack of egg trays. tall cardboard stack."
    
    review_queue = []
    total_images = 0
    total_annotations = 0
    
    image_paths = glob.glob(os.path.join(images_dir, '**', '*.*'), recursive=True)
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_paths)} images to process using Grounding DINO.")
    
    for img_path in image_paths:
        total_images += 1
        
        rel_path = os.path.relpath(img_path, images_dir)
        base, _ = os.path.splitext(rel_path)
        label_path = os.path.join(labels_dir, base + '.txt')
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open {img_path}: {e}")
            continue
            
        w, h = image.size
        
        # Prepare inputs
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Post-process
        # Using a low threshold to capture potential stacks, we will filter later if needed
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.20,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )[0]
        
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        
        if len(boxes) == 0:
            with open(label_path, 'w') as f:
                pass
            review_queue.append({"image": rel_path, "reason": "0 detections"})
            continue
            
        max_score = float(scores.max()) if len(scores) > 0 else 0.0
        
        with open(label_path, 'w') as f:
            for box in boxes:
                # Grounding DINO outputs [xmin, ymin, xmax, ymax] absolute pixels
                xmin, ymin, xmax, ymax = box
                # Convert to YOLO format: [x_center, y_center, width, height] normalized
                x_center = ((xmin + xmax) / 2.0) / w
                y_center = ((ymin + ymax) / 2.0) / h
                box_w = (xmax - xmin) / w
                box_h = (ymax - ymin) / h
                
                # Clip bounds
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                box_w = max(0.0, min(1.0, box_w))
                box_h = max(0.0, min(1.0, box_h))
                
                f.write(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
                total_annotations += 1
                
        if max_score < 0.40:
            review_queue.append({"image": rel_path, "reason": f"Low max confidence ({max_score:.2f} < 0.40)"})
            
    # Save review queue
    review_queue_path = os.path.join(dataset_dir, 'review_queue.json')
    with open(review_queue_path, 'w') as f:
        json.dump(review_queue, f, indent=4)
        
    print("\n========================================")
    print("Grounding DINO Auto-Annotation Report")
    print("========================================")
    print(f"Total Images Processed: {total_images}")
    print(f"Total Stack Annotations Generated: {total_annotations}")
    print(f"Images in Review Queue: {len(review_queue)}")
    print(f"Successfully Auto-Annotated (High Conf): {total_images - len(review_queue)}")
    print("========================================")

if __name__ == "__main__":
    main()
