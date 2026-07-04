import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class Evaluator:
    def __init__(self, data_dir="dataset_pilot"):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "images")
        self.anno_path = os.path.join(data_dir, "pilot_annotations.json")
        self.debug_dir = os.path.join(data_dir, "debug")
        os.makedirs(self.debug_dir, exist_ok=True)
        
        with open(self.anno_path, 'r') as f:
            self.annotations = json.load(f)
            
        # CNN Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_model = models.resnet18(pretrained=True)
        # Freeze backbone
        for param in self.cnn_model.parameters():
            param.requires_grad = False
        # Replace head for regression
        self.cnn_model.fc = nn.Linear(self.cnn_model.fc.in_features, 1)
        self.cnn_model = self.cnn_model.to(self.device)
        self.cnn_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def method_a_sobel(self, crop):
        h, w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel_y = np.absolute(sobel_y)
        edge_map = np.uint8(255 * sobel_y / np.max(sobel_y)) if np.max(sobel_y) > 0 else sobel_y
        
        profile = np.sum(edge_map, axis=1)
        window_size = max(3, int(h * 0.02))
        profile_smoothed = np.convolve(profile, np.ones(window_size)/window_size, mode='same')
        
        min_dist = max(1, int(h * 0.05))
        prominence = 5 * (np.mean(profile_smoothed) / 1000.0 + 1)
        peaks, _ = find_peaks(profile_smoothed, distance=min_dist, prominence=prominence)
        return len(peaks), edge_map, profile_smoothed, peaks

    def method_b_canny(self, crop):
        h, w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edge_map = cv2.Canny(blurred, 50, 150)
        
        profile = np.sum(edge_map, axis=1)
        window_size = max(3, int(h * 0.02))
        profile_smoothed = np.convolve(profile, np.ones(window_size)/window_size, mode='same')
        
        min_dist = max(1, int(h * 0.05))
        prominence = 5 * (np.mean(profile_smoothed) / 1000.0 + 1)
        peaks, _ = find_peaks(profile_smoothed, distance=min_dist, prominence=prominence)
        return len(peaks), edge_map, profile_smoothed, peaks

    def method_c_morphological(self, crop):
        h, w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Horizontal structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        edge_map = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        profile = np.sum(edge_map, axis=1)
        window_size = max(3, int(h * 0.02))
        profile_smoothed = np.convolve(profile, np.ones(window_size)/window_size, mode='same')
        
        min_dist = max(1, int(h * 0.05))
        prominence = 5 * (np.mean(profile_smoothed) / 1000.0 + 1)
        peaks, _ = find_peaks(profile_smoothed, distance=min_dist, prominence=prominence)
        return len(peaks), edge_map, profile_smoothed, peaks

    def method_d_cnn(self, crop):
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = self.cnn_transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.cnn_model.eval()
            output = self.cnn_model(tensor)
        return max(0, int(round(output.item()))), None, None, None

    def create_debug_strip(self, crop, out_a, out_b, out_c, save_path):
        fig, axes = plt.subplots(1, 4, figsize=(16, 8))
        
        axes[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Crop")
        
        # Method A
        axes[1].imshow(out_a[1], cmap='gray')
        for p in out_a[3]:
            axes[1].axhline(y=p, color='r', linestyle='-', alpha=0.5)
        axes[1].set_title(f"Sobel (Count: {out_a[0]})")
        
        # Method B
        axes[2].imshow(out_b[1], cmap='gray')
        for p in out_b[3]:
            axes[2].axhline(y=p, color='r', linestyle='-', alpha=0.5)
        axes[2].set_title(f"Canny (Count: {out_b[0]})")
        
        # Method C
        axes[3].imshow(out_c[1], cmap='gray')
        for p in out_c[3]:
            axes[3].axhline(y=p, color='r', linestyle='-', alpha=0.5)
        axes[3].set_title(f"Morph (Count: {out_c[0]})")
        
        for ax in axes:
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def run_evaluation(self):
        results = {"A": [], "B": [], "C": [], "D": []}
        
        print("Running evaluation on Pilot Annotations...")
        
        for img_name, stacks in self.annotations.items():
            img_path = os.path.join(self.img_dir, img_name)
            if not os.path.exists(img_path):
                continue
                
            image = cv2.imread(img_path)
            h, w = image.shape[:2]
            
            for i, stack in enumerate(stacks):
                xc, yc, bw, bh = stack["bbox_yolo"]
                true_count = stack["true_count"]
                
                # YOLO format to absolute coords
                x1 = max(0, int((xc - bw/2) * w))
                y1 = max(0, int((yc - bh/2) * h))
                x2 = min(w, int((xc + bw/2) * w))
                y2 = min(h, int((yc + bh/2) * h))
                
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                    
                # Run methods
                out_a = self.method_a_sobel(crop)
                out_b = self.method_b_canny(crop)
                out_c = self.method_morphological(crop) if hasattr(self, 'method_morphological') else self.method_c_morphological(crop)
                out_d = self.method_d_cnn(crop)
                
                results["A"].append((out_a[0], true_count))
                results["B"].append((out_b[0], true_count))
                results["C"].append((out_c[0], true_count))
                results["D"].append((out_d[0], true_count))
                
                # Debug output
                debug_path = os.path.join(self.debug_dir, f"{img_name.split('.')[0]}_stack_{i}.png")
                self.create_debug_strip(crop, out_a, out_b, out_c, debug_path)

        self.print_metrics(results)

    def print_metrics(self, results):
        print("\n==========================================")
        print("PILOT STUDY RESULTS: Counting Methodology")
        print("==========================================")
        
        methods = {"A": "Sobel Projection", "B": "Canny Edge", "C": "Morph Top-Hat", "D": "CNN Regression"}
        
        for key in ["A", "B", "C", "D"]:
            preds = np.array([r[0] for r in results[key]])
            trues = np.array([r[1] for r in results[key]])
            
            if len(trues) == 0:
                print(f"{methods[key]}: No data.")
                continue
                
            mae = np.mean(np.abs(preds - trues))
            rmse = np.sqrt(np.mean((preds - trues)**2))
            
            # Avoid division by zero for MAPE
            safe_trues = np.where(trues == 0, 1e-6, trues)
            mape = np.mean(np.abs((preds - trues) / safe_trues)) * 100
            
            print(f"Method: {methods[key]}")
            print(f"  MAE:  {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAPE: {mape:.2f}%")
            print("-" * 40)
            
if __name__ == "__main__":
    pilot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset_pilot'))
    evaluator = Evaluator(data_dir=pilot_dir)
    evaluator.run_evaluation()
