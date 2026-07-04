import os
import shutil
import random
import glob
from collections import defaultdict
import yaml

def prepare_dataset(source_dir="eggcounterdataset", dest_dir="dataset_processed", seed=42):
    print(f"Starting dataset preparation from '{source_dir}' to '{dest_dir}'...")
    print(f"Using random seed: {seed}")
    
    random.seed(seed)
    
    # 1. Inspect the Dataset
    image_extensions = ('.jpg', '.jpeg', '.png')
    
    # Gather all files
    all_files = os.listdir(source_dir) if os.path.exists(source_dir) else []
    
    images = {}
    labels = {}
    corrupted_images = []
    
    for filename in all_files:
        filepath = os.path.join(source_dir, filename)
        if not os.path.isfile(filepath):
            continue
            
        name, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        if ext in image_extensions:
            # Simple corruption check: file size > 0
            if os.path.getsize(filepath) > 0:
                images[name] = filename
            else:
                corrupted_images.append(filename)
        elif ext == '.txt' and name != 'classes':
            labels[name] = filename

    # Find orphans and duplicates
    # Since we use a dict with `name` as key, duplicates with exact same name+ext won't be seen by listdir natively in Windows,
    # but if they have different extensions (e.g. img.jpg and img.png), the dict overwrites. Let's assume unique names.
    
    paired_names = set(images.keys()).intersection(set(labels.keys()))
    orphan_images = set(images.keys()) - set(labels.keys())
    orphan_labels = set(labels.keys()) - set(images.keys())
    
    # 2. Validate Labels
    valid_labels = 0
    invalid_labels = 0
    classes_found = set()
    
    for name in paired_names | orphan_labels:
        label_file = labels[name]
        filepath = os.path.join(source_dir, label_file)
        is_valid = True
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue # Empty row
                    if len(parts) != 5:
                        is_valid = False
                        break
                    
                    cls_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:5])
                    
                    if not (0 <= x_c <= 1 and 0 <= y_c <= 1):
                        is_valid = False
                        break
                    if not (0 < w <= 1 and 0 < h <= 1):
                        is_valid = False
                        break
                        
                    classes_found.add(cls_id)
        except Exception:
            is_valid = False
            
        if is_valid:
            valid_labels += 1
        else:
            invalid_labels += 1
            # We could remove invalid labels from processing, but instructions say "Do not silently ignore problems."
            print(f"Warning: Invalid label file found: {label_file}")
            
    # For splitting, we only split the pairs, or all images?
    # Instructions: "Every image must always stay paired with its matching YOLO label. Never separate matching pairs."
    # What about images without labels? YOLO allows background images (images without label files) to be used as negative samples.
    # So we should split all valid images. If an image has a label, we copy it.
    
    valid_image_names = list(images.keys())
    # Sort for reproducibility before random shuffle
    valid_image_names.sort()
    random.shuffle(valid_image_names)
    
    total_images = len(valid_image_names)
    
    # 3. Split the Dataset (70 / 20 / 10)
    train_count = int(total_images * 0.7)
    val_count = int(total_images * 0.2)
    # test gets the remainder
    
    splits = {
        'train': valid_image_names[:train_count],
        'val': valid_image_names[train_count:train_count+val_count],
        'test': valid_image_names[train_count+val_count:]
    }
    
    # Safety Check for small datasets
    if total_images < 200:
        print(f"Note: Dataset is small ({total_images} images). A different split like 80/10/10 or cross-validation might be better, but keeping default 70/20/10 as requested.")

    # 4 & 8. Create Structure and Copy Files Safely
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir) # Clean if exists
        
    for split_name in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, 'images', split_name), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'labels', split_name), exist_ok=True)
        
    for split_name, item_names in splits.items():
        for name in item_names:
            # Copy image
            img_filename = images[name]
            src_img = os.path.join(source_dir, img_filename)
            dst_img = os.path.join(dest_dir, 'images', split_name, img_filename)
            shutil.copy2(src_img, dst_img)
            
            # Copy label if exists
            if name in labels:
                lbl_filename = labels[name]
                src_lbl = os.path.join(source_dir, lbl_filename)
                dst_lbl = os.path.join(dest_dir, 'labels', split_name, lbl_filename)
                shutil.copy2(src_lbl, dst_lbl)
                
    # 5. Generate data.yaml
    # If no classes found (no labels), default to 0: stack
    if not classes_found:
        classes_dict = {0: 'stack'}
    else:
        classes_dict = {i: f'class_{i}' for i in sorted(list(classes_found))}
        
    yaml_content = {
        'path': os.path.abspath(dest_dir).replace('\\', '/'),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': classes_dict
    }
    
    yaml_path = os.path.join(dest_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    # 6 & 10. Final Output
    total_annotations = valid_labels # In our count, valid_labels is count of valid label *files*, not total bounding boxes.
    # To get total annotations (bounding boxes):
    total_bboxes = 0
    for name in paired_names:
        filepath = os.path.join(source_dir, labels[name])
        try:
            with open(filepath, 'r') as f:
                total_bboxes += len([line for line in f.readlines() if line.strip()])
        except:
            pass
            
    avg_objects = total_bboxes / total_images if total_images > 0 else 0

    print("\n========================================")
    print("Dataset Preparation Complete")
    print("========================================")
    print(f"Total Images: {total_images}")
    print(f"Train: {len(splits['train'])}")
    print(f"Validation: {len(splits['val'])}")
    print(f"Test: {len(splits['test'])}")
    print(f"Labels Verified: {valid_labels}")
    print(f"Missing Labels (Images without annotations): {len(orphan_images)}")
    print(f"Orphan Labels (Labels without images): {len(orphan_labels)}")
    print(f"Corrupted Files: {len(corrupted_images) + invalid_labels}")
    print(f"Total Annotations (Bounding Boxes): {total_bboxes}")
    print(f"Average Objects per Image: {avg_objects:.2f}")
    print(f"\nOutput Directory:\n{os.path.abspath(dest_dir)}")
    print("========================================\n")

if __name__ == "__main__":
    # Ensure source directory exists
    source = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eggcounterdataset'))
    dest = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset_processed'))
    
    prepare_dataset(source_dir=source, dest_dir=dest, seed=42)
