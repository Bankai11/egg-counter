import os
import random
import cv2
import numpy as np

def create_egg_tray_image(width, height, num_trays):
    # Create background (wood-like or concrete-like color)
    bg_color = (random.randint(40, 100), random.randint(50, 110), random.randint(60, 120))
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    # Add some background noise / texture
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    boxes = []
    
    # Try to place multiple trays
    attempts = 0
    placed = 0
    
    while placed < num_trays and attempts < 100:
        attempts += 1
        
        # Random dimensions for the tray
        tray_w = random.randint(150, 280)
        tray_h = random.randint(150, 280)
        
        # Random position
        x_min = random.randint(10, width - tray_w - 10)
        y_min = random.randint(10, height - tray_h - 10)
        
        x_max = x_min + tray_w
        y_max = y_min + tray_h
        
        # Check overlap with already placed trays
        overlap = False
        for (bx1, by1, bx2, by2) in boxes:
            # Calculate intersection
            ix1 = max(x_min, bx1)
            iy1 = max(y_min, by1)
            ix2 = min(x_max, bx2)
            iy2 = min(y_max, by2)
            
            if ix1 < ix2 and iy1 < iy2:
                # Calculate intersection area / smaller tray area
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                smaller_area = min(tray_w * tray_h, (bx2 - bx1) * (by2 - by1))
                if intersection_area / smaller_area > 0.15:  # Allow max 15% overlap
                    overlap = True
                    break
        
        if overlap:
            continue
            
        # Draw the tray
        # Trays can be different colors (gray paper pulp, blue/red/green/yellow plastic)
        tray_type = random.choice(['pulp', 'plastic_blue', 'plastic_red', 'plastic_green'])
        if tray_type == 'pulp':
            tray_color = (random.randint(160, 200), random.randint(160, 200), random.randint(150, 190))  # Beige-gray
        elif tray_type == 'plastic_blue':
            tray_color = (random.randint(180, 220), random.randint(100, 140), random.randint(20, 60))    # Blueish (BGR)
        elif tray_type == 'plastic_red':
            tray_color = (random.randint(20, 60), random.randint(30, 70), random.randint(180, 230))     # Reddish (BGR)
        else:
            tray_color = (random.randint(30, 70), random.randint(150, 200), random.randint(30, 70))     # Greenish (BGR)
            
        # Draw main tray card
        # Make corners slightly rounded
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), tray_color, -1)
        # Draw dark border
        border_color = (int(tray_color[0]*0.7), int(tray_color[1]*0.7), int(tray_color[2]*0.7))
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), border_color, 2)
        
        # Draw tray grid (e.g. 5x6 or 4x4 slots)
        rows = random.choice([5, 6])
        cols = random.choice([5, 6])
        
        cell_w = tray_w / cols
        cell_h = tray_h / rows
        
        # Draw slots and eggs
        for r in range(rows):
            for c in range(cols):
                slot_cx = int(x_min + (c + 0.5) * cell_w)
                slot_cy = int(y_min + (r + 0.5) * cell_h)
                slot_r = int(min(cell_w, cell_h) * 0.35)
                
                # Draw empty slot base circle
                cv2.circle(img, (slot_cx, slot_cy), slot_r, border_color, 1)
                
                # Decide if slot contains an egg (75% chance)
                if random.random() < 0.75:
                    # Egg color: white or brown
                    if random.random() < 0.5:
                        egg_color = (random.randint(220, 245), random.randint(225, 250), random.randint(230, 255)) # White egg
                    else:
                        egg_color = (random.randint(120, 150), random.randint(150, 180), random.randint(180, 210)) # Brown egg (BGR)
                    
                    # Draw shadow for egg
                    cv2.circle(img, (slot_cx + 2, slot_cy + 2), int(slot_r * 0.95), (0, 0, 0), -1)
                    # Draw egg body
                    cv2.circle(img, (slot_cx, slot_cy), int(slot_r * 0.95), egg_color, -1)
                    # Draw highlight
                    cv2.circle(img, (slot_cx - 2, slot_cy - 2), int(slot_r * 0.25), (255, 255, 255), -1)
                else:
                    # Empty slot shadow
                    cv2.circle(img, (slot_cx, slot_cy), int(slot_r * 0.7), (int(border_color[0]*0.5), int(border_color[1]*0.5), int(border_color[2]*0.5)), -1)
        
        boxes.append((x_min, y_min, x_max, y_max))
        placed += 1
        
    return img, boxes

def save_yolo_dataset(base_dir, split, index, img, boxes):
    # Create directories
    images_dir = os.path.join(base_dir, split, 'images')
    labels_dir = os.path.join(base_dir, split, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Save Image
    img_name = f'tray_{index:04d}.jpg'
    img_path = os.path.join(images_dir, img_name)
    cv2.imwrite(img_path, img)
    
    # Save Label
    lbl_name = f'tray_{index:04d}.txt'
    lbl_path = os.path.join(labels_dir, lbl_name)
    
    h, w, _ = img.shape
    
    with open(lbl_path, 'w') as f:
        for (x1, y1, x2, y2) in boxes:
            # Convert to YOLO format: class x_center y_center width height (normalized)
            dw = 1.0 / w
            dh = 1.0 / h
            x_center = (x1 + x2) / 2.0 * dw
            y_center = (y1 + y2) / 2.0 * dh
            width_val = (x2 - x1) * dw
            height_val = (y2 - y1) * dh
            
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width_val:.6f} {height_val:.6f}\n")

def main():
    random.seed(42)
    np.random.seed(42)
    
    base_dir = './dataset'
    
    # Splits configuration
    splits = {
        'train': 50,
        'val': 15,
        'test': 10
    }
    
    total_generated = 0
    print("Generating synthetic egg tray dataset...")
    
    for split, count in splits.items():
        print(f"Generating {count} images for split '{split}'...")
        for i in range(count):
            num_trays = random.randint(1, 3)
            # Create a 640x640 image
            img, boxes = create_egg_tray_image(640, 640, num_trays)
            save_yolo_dataset(base_dir, split, total_generated, img, boxes)
            total_generated += 1
            
    print(f"Dataset generation complete! Total images: {total_generated}")
    print(f"Dataset path: {os.path.abspath(base_dir)}")

if __name__ == '__main__':
    main()
