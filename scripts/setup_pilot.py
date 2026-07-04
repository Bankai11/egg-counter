import os
import random
import shutil
import json

def setup_pilot(source_dir, pilot_dir, num_samples=20, seed=42):
    random.seed(seed)
    
    source_images_dir = os.path.join(source_dir, "images", "train")
    pilot_images_dir = os.path.join(pilot_dir, "images")
    
    if not os.path.exists(source_images_dir):
        print(f"Error: Source directory {source_images_dir} does not exist.")
        return
        
    os.makedirs(pilot_images_dir, exist_ok=True)
    
    all_images = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(all_images) < num_samples:
        print(f"Warning: Only found {len(all_images)} images, taking all of them.")
        sampled_images = all_images
    else:
        sampled_images = random.sample(all_images, num_samples)
        
    annotations = {}
    
    for img_name in sampled_images:
        src_path = os.path.join(source_images_dir, img_name)
        dst_path = os.path.join(pilot_images_dir, img_name)
        shutil.copy2(src_path, dst_path)
        
        # Template for annotations
        annotations[img_name] = [
            {
                "bbox_yolo": [0.5, 0.5, 0.2, 0.8], # Placeholder [x_center, y_center, width, height]
                "true_count": 0 # Placeholder count
            }
        ]
        
    annotations_path = os.path.join(pilot_dir, "pilot_annotations.json")
    with open(annotations_path, 'w') as f:
        json.dump(annotations, f, indent=4)
        
    print(f"Pilot study setup complete.")
    print(f"Sampled {len(sampled_images)} images into {pilot_images_dir}")
    print(f"Generated annotation template at {annotations_path}")

if __name__ == "__main__":
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset_processed'))
    pilot = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset_pilot'))
    setup_pilot(src, pilot)
