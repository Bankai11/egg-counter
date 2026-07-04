import os
import glob
import random
import cv2
import matplotlib.pyplot as plt

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_dir = os.path.join(root_dir, 'dataset_processed')
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    audit_dir = os.path.join(dataset_dir, 'audit')
    os.makedirs(audit_dir, exist_ok=True)
    
    # Find all images
    image_paths = glob.glob(os.path.join(images_dir, '**', '*.*'), recursive=True)
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        print("No images found.")
        return
        
    # Sample 30 images (or all if < 30)
    sample_size = min(30, len(image_paths))
    sampled_paths = random.sample(image_paths, sample_size)
    
    print(f"Auditing {sample_size} images...")
    
    for i, img_path in enumerate(sampled_paths):
        # Determine corresponding label path
        rel_path = os.path.relpath(img_path, images_dir)
        base, _ = os.path.splitext(rel_path)
        label_path = os.path.join(labels_dir, base + '.txt')
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Read labels
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        x_c, y_c, bw, bh = map(float, parts[1:5])
                        
                        # Unnormalize
                        x1 = int((x_c - bw/2) * w)
                        y1 = int((y_c - bh/2) * h)
                        x2 = int((x_c + bw/2) * w)
                        y2 = int((y_c + bh/2) * h)
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(img, "YOLOv8 Auto", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
        # Save output
        out_name = f"audit_{i:02d}_{os.path.basename(img_path)}"
        out_path = os.path.join(audit_dir, out_name)
        cv2.imwrite(out_path, img)
        
    print(f"Audit complete. Generated {sample_size} visualization images in {audit_dir}")

if __name__ == "__main__":
    main()
