import os
import torch
from ultralytics import YOLO

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_yaml = os.path.join(root_dir, 'eggtrainset', 'data.yaml')
    
    if not os.path.exists(data_yaml):
        print(f"Error: Could not find {data_yaml}")
        return

    # Load YOLO11m model (will download if not found)
    print("Initializing YOLO11m...")
    model = YOLO("yolo11m.pt")
    
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    
    # Train the model
    # Note: We use a moderate number of epochs and patience for early stopping
    print("Starting training...")
    results = model.train(
        data=data_yaml,
        epochs=300, # Increased for production-grade accuracy
        imgsz=640,
        batch=4, # Reduced batch size for stability
        name="yolo11m_egg_tray",
        project="runs/detect",
        device=device,
        save=True,
        exist_ok=True,
        verbose=True
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()
