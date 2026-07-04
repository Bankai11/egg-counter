import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def parse_yolo_label(file_path, img_w, img_h):
    boxes = []
    if not os.path.exists(file_path):
        return boxes
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                x_c = float(parts[1]) * img_w
                y_c = float(parts[2]) * img_h
                w = float(parts[3]) * img_w
                h = float(parts[4]) * img_h
                boxes.append((cls_id, x_c, y_c, w, h))
    return boxes

def run_eda(dataset_path):
    images_path = os.path.join(dataset_path, 'train', 'images', '*.jpg')
    image_files = glob.glob(images_path)
    
    if not image_files:
        print(f"No images found in {images_path}")
        return

    print(f"Found {len(image_files)} training images.")
    
    hsv_means = []
    hsv_stds = []
    widths = []
    heights = []
    aspect_ratios = []
    small_boxes = 0
    extreme_ar_boxes = 0

    for img_file in image_files:
        img = cv2.imread(img_file)
        if img is None:
            continue
            
        img_h, img_w, _ = img.shape
        
        # Lighting Variation
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_hsv = np.mean(hsv_img, axis=(0, 1))
        std_hsv = np.std(hsv_img, axis=(0, 1))
        
        hsv_means.append(mean_hsv)
        hsv_stds.append(std_hsv)
        
        # Labels
        label_file = img_file.replace('images', 'labels').replace('.jpg', '.txt')
        boxes = parse_yolo_label(label_file, img_w, img_h)
        
        for (cls_id, x_c, y_c, w, h) in boxes:
            widths.append(w)
            heights.append(h)
            ar = w / h if h > 0 else 0
            aspect_ratios.append(ar)
            
            # Sanity Checks
            if w < 10 or h < 10:
                small_boxes += 1
            if ar > 10 or ar < 0.1:
                extreme_ar_boxes += 1
                
    # Aggregate stats
    avg_mean_hsv = np.mean(hsv_means, axis=0)
    avg_std_hsv = np.mean(hsv_stds, axis=0)
    
    print("\n--- EDA RESULTS ---")
    print(f"Average HSV Mean (H, S, V): {avg_mean_hsv}")
    print(f"Average HSV Std  (H, S, V): {avg_std_hsv}")
    print(f"Total Bounding Boxes: {len(widths)}")
    print(f"Average Box Width: {np.mean(widths):.2f} px")
    print(f"Average Box Height: {np.mean(heights):.2f} px")
    print(f"Average Aspect Ratio: {np.mean(aspect_ratios):.2f}")
    
    print("\n--- SANITY CHECKS ---")
    print(f"Boxes too small (<10px): {small_boxes}")
    print(f"Boxes with extreme aspect ratio (>10 or <0.1): {extreme_ar_boxes}")

    # Plotting
    plt.figure(figsize=(15, 5))
    
    # 1. Width vs Height Scatter
    plt.subplot(1, 3, 1)
    plt.scatter(widths, heights, alpha=0.5)
    plt.title("Box Width vs Height")
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    
    # 2. Aspect Ratio Histogram
    plt.subplot(1, 3, 2)
    plt.hist(aspect_ratios, bins=50, color='blue', alpha=0.7)
    plt.title("Aspect Ratio Distribution")
    plt.xlabel("Aspect Ratio (W/H)")
    plt.ylabel("Frequency")
    
    # 3. HSV V-channel (Brightness) Distribution
    v_means = [m[2] for m in hsv_means]
    plt.subplot(1, 3, 3)
    plt.hist(v_means, bins=30, color='orange', alpha=0.7)
    plt.title("Image Brightness (V-channel) Distribution")
    plt.xlabel("Mean V Value")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plot_path = os.path.join(dataset_path, 'eda_plots.png')
    plt.savefig(plot_path)
    print(f"\nPlots saved to {plot_path}")

if __name__ == '__main__':
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    run_eda(dataset_dir)
