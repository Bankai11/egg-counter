import argparse
import sys
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for Egg Tray Detection")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="", help="Device (cpu, cuda, or 0, 1 etc.)")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Pretrained model weights to start from")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loader workers")
    
    args = parser.parse_args()
    
    print(f"Loading weights: {args.weights}...")
    model = YOLO(args.weights)
    
    # Run training
    print(f"Starting training on dataset {args.data} for {args.epochs} epochs...")
    
    # We specify a simple call. Ultralytics automatically handles device matching if args.device is empty.
    # Note: we use standard augmentations.
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        workers=args.workers,
        mosaic=1.0,       # Mosaic augmentation (0.0 to 1.0)
        hsv_h=0.015,      # HSV color jitter (hue)
        hsv_s=0.7,        # HSV color jitter (saturation)
        hsv_v=0.4,        # HSV color jitter (value)
        degrees=10.0,     # Random rotation degrees
        translate=0.1,    # Random translation
        scale=0.5,        # Random scale factor
        fliplr=0.5,       # Left-right flips
        flipud=0.0,       # Up-down flips
        project="egg_tray_training",
        name="yolov8_egg_tray"
    )
    
    print("Training finished!")
    print("Best weights saved under egg_tray_training/yolov8_egg_tray/weights/best.pt")

if __name__ == "__main__":
    main()
