import os
import glob
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

def calculate_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def yolo_to_xyxy(x_center, y_center, w, h, img_w, img_h):
    x1 = (x_center - w / 2) * img_w
    y1 = (y_center - h / 2) * img_h
    x2 = (x_center + w / 2) * img_w
    y2 = (y_center + h / 2) * img_h
    return [x1, y1, x2, y2]

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    images_dir = os.path.join(root_dir, 'eggtrainset', 'test', 'images')
    labels_dir = os.path.join(root_dir, 'eggtrainset', 'test', 'labels')
    weights_path = r"C:\Users\banka\runs\detect\runs\detect\yolo11m_egg_tray\weights\best.pt"
    
    out_dir = os.path.join(root_dir, 'reports', 'error_analysis')
    fn_dir = os.path.join(out_dir, 'false_negatives')
    fp_dir = os.path.join(out_dir, 'false_positives')
    
    os.makedirs(fn_dir, exist_ok=True)
    os.makedirs(fp_dir, exist_ok=True)

    print(f"Loading model from {weights_path}")
    model = YOLO(weights_path)
    
    # Store stats
    all_confs = []
    tp_confs = []
    fp_confs = []
    
    gt_widths = []
    gt_heights = []
    pred_widths = []
    pred_heights = []
    
    pred_centers_x = []
    pred_centers_y = []
    
    image_scores = []
    
    image_files = glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.jpeg"))
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(labels_dir, base_name + ".txt")
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_h, img_w = img.shape[:2]
        
        # Load GT
        gt_boxes = [] # [[x1, y1, x2, y2]]
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c, xc, yc, w, h = map(float, parts[:5])
                        box = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
                        gt_boxes.append(box)
                        gt_widths.append(w * img_w)
                        gt_heights.append(h * img_h)
                        
        # Predict
        results = model.predict(img_path, verbose=False, conf=0.1) # Lower conf to see more
        pred_boxes = [] # [[x1,y1,x2,y2,conf]]
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                pred_boxes.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf])
                
                # For plotting
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                pred_widths.append(w)
                pred_heights.append(h)
                
                xc = xyxy[0] + w/2
                yc = xyxy[1] + h/2
                # Normalized centers for heatmap
                pred_centers_x.append(xc / img_w)
                pred_centers_y.append(yc / img_h)
                all_confs.append(conf)
                
        # Matching
        matched_gt = set()
        matched_pred = set()
        
        # Sort predictions by confidence desc
        pred_boxes.sort(key=lambda x: x[4], reverse=True)
        
        for i, pred in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            for j, gt in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = calculate_iou(pred[:4], gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= 0.5:
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)
                tp_confs.append(pred[4])
            else:
                fp_confs.append(pred[4])
                
        # Find FNs and FPs
        fns = [gt for j, gt in enumerate(gt_boxes) if j not in matched_gt]
        fps = [pred for i, pred in enumerate(pred_boxes) if i not in matched_pred]
        
        # Image score (avg confidence of detections, if any)
        avg_conf = np.mean([p[4] for p in pred_boxes]) if pred_boxes else 0
        image_scores.append({'name': img_name, 'avg_conf': avg_conf, 'fns': len(fns), 'fps': len(fps)})
        
        # Save FN overlays
        if len(fns) > 0:
            fn_img = img.copy()
            for fn in fns:
                cv2.rectangle(fn_img, (int(fn[0]), int(fn[1])), (int(fn[2]), int(fn[3])), (0, 0, 255), 3)
                cv2.putText(fn_img, "FN (Missed)", (int(fn[0]), int(fn[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(fn_dir, img_name), fn_img)
            
        # Save FP overlays
        if len(fps) > 0:
            fp_img = img.copy()
            for fp in fps:
                cv2.rectangle(fp_img, (int(fp[0]), int(fp[1])), (int(fp[2]), int(fp[3])), (255, 0, 0), 3)
                cv2.putText(fp_img, f"FP ({fp[4]:.2f})", (int(fp[0]), int(fp[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.imwrite(os.path.join(fp_dir, img_name), fp_img)

    # 1. Confidence Histogram
    plt.figure(figsize=(10,6))
    if len(tp_confs) > 0:
        plt.hist(tp_confs, bins=20, alpha=0.5, label='True Positives', color='green')
    if len(fp_confs) > 0:
        plt.hist(fp_confs, bins=20, alpha=0.5, label='False Positives', color='red')
    plt.title('Confidence Distribution (TP vs FP)')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'confidence_histogram.png'))
    plt.close()
    
    # 2. BBox Size Distribution
    plt.figure(figsize=(10,6))
    if len(gt_widths) > 0:
        plt.scatter(gt_widths, gt_heights, alpha=0.3, label='Ground Truth', color='blue')
    if len(pred_widths) > 0:
        plt.scatter(pred_widths, pred_heights, alpha=0.3, label='Predictions', color='orange')
    plt.title('Bounding Box Size Distribution')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'bbox_sizes.png'))
    plt.close()
    
    # 3. Heatmap
    if len(pred_centers_x) > 0:
        plt.figure(figsize=(10,8))
        sns.kdeplot(x=pred_centers_x, y=pred_centers_y, cmap="Reds", fill=True, bw_adjust=0.5)
        plt.xlim(0, 1)
        plt.ylim(1, 0) # y-axis inverted for image coords
        plt.title('Detection Center Heatmap (Normalized)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.savefig(os.path.join(out_dir, 'detection_heatmap.png'))
        plt.close()
        
    # Copy PR curve
    val_dir = r"C:\Users\banka\runs\detect\val-2"
    pr_curve = os.path.join(val_dir, "BoxPR_curve.png")
    if os.path.exists(pr_curve):
        shutil.copy(pr_curve, os.path.join(out_dir, 'PR_curve.png'))
        
    # Worst images list
    image_scores.sort(key=lambda x: x['avg_conf'])
    with open(os.path.join(out_dir, 'worst_images.txt'), 'w') as f:
        f.write("Images sorted by lowest average confidence of detections:\n")
        f.write("Format: [Image Name] - Avg Conf: X.XX | FNs: Y | FPs: Z\n\n")
        for sc in image_scores[:20]:
            f.write(f"{sc['name']} - Avg Conf: {sc['avg_conf']:.2f} | FNs: {sc['fns']} | FPs: {sc['fps']}\n")
            
    print("Failure analysis complete! Outputs saved to reports/error_analysis")

if __name__ == "__main__":
    main()
