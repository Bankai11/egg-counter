import os
import glob
from collections import defaultdict

def validate_labels(dataset_dir="dataset_processed", valid_classes={0}):
    labels_dir = os.path.join(dataset_dir, "labels")
    
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory not found at {labels_dir}")
        return

    txt_files = glob.glob(os.path.join(labels_dir, "**", "*.txt"), recursive=True)
    
    if not txt_files:
        print(f"No label (.txt) files found in {labels_dir}")
        return
        
    total_files = len(txt_files)
    empty_files = []
    malformed_files = defaultdict(list)
    out_of_bounds_files = defaultdict(list)
    invalid_class_files = defaultdict(list)
    
    valid_file_count = 0
    total_annotations = 0

    print(f"Starting validation on {total_files} label files in '{labels_dir}'...")

    for filepath in txt_files:
        # Ignore classes.txt if present
        if os.path.basename(filepath) == 'classes.txt':
            continue
            
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            empty_files.append(filepath)
            valid_file_count += 1  # Empty files are technically valid YOLO background images
            continue
            
        file_is_valid = True
        
        for row_idx, line in enumerate(lines):
            parts = line.strip().split()
            if not parts:
                continue
                
            if len(parts) != 5:
                malformed_files[filepath].append(f"Row {row_idx+1}: Expected 5 values, got {len(parts)} -> '{line.strip()}'")
                file_is_valid = False
                continue
                
            try:
                cls_id = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:5])
                
                if cls_id not in valid_classes:
                    invalid_class_files[filepath].append(f"Row {row_idx+1}: Invalid class ID {cls_id}")
                    file_is_valid = False
                    
                if not (0.0 <= x_c <= 1.0 and 0.0 <= y_c <= 1.0):
                    out_of_bounds_files[filepath].append(f"Row {row_idx+1}: Center coords (x,y) out of bounds [0, 1] -> ({x_c}, {y_c})")
                    file_is_valid = False
                    
                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    out_of_bounds_files[filepath].append(f"Row {row_idx+1}: Width/Height out of bounds (0, 1] -> ({w}, {h})")
                    file_is_valid = False
                    
                if file_is_valid:
                    total_annotations += 1
                    
            except ValueError as e:
                malformed_files[filepath].append(f"Row {row_idx+1}: Cannot parse values as floats -> {e}")
                file_is_valid = False
                
        if file_is_valid:
            valid_file_count += 1

    print("\n========================================")
    print("Label Validation Report")
    print("========================================")
    print(f"Total Files Scanned: {total_files}")
    print(f"Total Valid Files: {valid_file_count}")
    print(f"Total Valid Annotations: {total_annotations}")
    print(f"Empty Files (Background Images): {len(empty_files)}")
    
    issues_found = False
    
    if malformed_files:
        issues_found = True
        print(f"\n[!] Malformed Files ({len(malformed_files)}):")
        for f, errors in malformed_files.items():
            print(f"  - {os.path.basename(f)}")
            for e in errors:
                print(f"      {e}")
                
    if out_of_bounds_files:
        issues_found = True
        print(f"\n[!] Out of Bounds Coordinates ({len(out_of_bounds_files)}):")
        for f, errors in out_of_bounds_files.items():
            print(f"  - {os.path.basename(f)}")
            for e in errors:
                print(f"      {e}")
                
    if invalid_class_files:
        issues_found = True
        print(f"\n[!] Invalid Class IDs ({len(invalid_class_files)}):")
        for f, errors in invalid_class_files.items():
            print(f"  - {os.path.basename(f)}")
            for e in errors:
                print(f"      {e}")
                
    print("\n========================================")
    if issues_found:
        print("STATUS: FAILED. Please fix the above issues before training.")
        return False
    else:
        print("STATUS: PASSED. All labels are formatted correctly.")
        return True

if __name__ == "__main__":
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset_processed'))
    # Expecting only class 0 ('stack')
    validate_labels(dataset_dir=dataset_path, valid_classes={0})
