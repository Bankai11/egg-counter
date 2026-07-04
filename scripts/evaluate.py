import os
import torch
from ultralytics import YOLO

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_yaml = os.path.join(root_dir, 'eggtrainset', 'data.yaml')
    
    # The best weights from our training run
    weights_path = r"C:\Users\banka\runs\detect\runs\detect\yolo11m_egg_tray\weights\best.pt"
    
    if not os.path.exists(weights_path):
        print(f"Error: Could not find trained weights at {weights_path}")
        return
            
    print(f"Loading trained model from {weights_path}")
    model = YOLO(weights_path)
    
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on device: {device}")
    
    # Run evaluation on the test set
    metrics = model.val(data=data_yaml, split='test', device=device)
    
    # Extract and format metrics
    map50 = metrics.box.map50
    map75 = metrics.box.map75
    map50_95 = metrics.box.map
    precision = metrics.box.p.mean()
    recall = metrics.box.r.mean()
    
    print("\n" + "="*50)
    print("FINAL MODEL EVALUATION RESULTS (Test Set)")
    print("="*50)
    print(f"mAP@0.5      : {map50:.4f}")
    print(f"mAP@0.75     : {map75:.4f}")
    print(f"mAP@0.5:0.95 : {map50_95:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print("="*50)
    
if __name__ == "__main__":
    main()
