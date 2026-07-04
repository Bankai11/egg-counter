import os
import yaml
from ultralytics import YOLO

def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'pipeline.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    model_config = config['model']
    aug_config = config['augmentations']
    
    # Initialize YOLO model
    # Note: Using yolo11m as planned. If weights aren't available, ultralytics will auto-download
    # But wait, in Ultralytics, YOLO11 uses the 'yolo11n.pt' nomenclature. 
    # I'll use 'yolo11m.pt'
    print(f"Loading model {model_config['name']}...")
    model = YOLO(model_config['name'])
    
    # Absolute path for data.yaml
    data_yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data.yaml'))
    
    print("Starting training...")
    # Train the model with specified hyperparams and augmentations
    results = model.train(
        data=data_yaml_path,
        epochs=model_config['epochs'],
        imgsz=model_config['imgsz'],
        batch=model_config['batch'],
        patience=model_config['patience'],
        optimizer=model_config['optimizer'],
        lr0=model_config['lr0'],
        lrf=model_config['lrf'],
        cos_lr=model_config['cos_lr'],
        
        # Augmentations
        hsv_h=aug_config['hsv_h'],
        hsv_s=aug_config['hsv_s'],
        hsv_v=aug_config['hsv_v'],
        degrees=aug_config['degrees'],
        translate=aug_config['translate'],
        scale=aug_config['scale'],
        shear=aug_config['shear'],
        perspective=aug_config['perspective'],
        flipud=aug_config['flipud'],
        fliplr=aug_config['fliplr'],
        mosaic=aug_config['mosaic'],
        mixup=aug_config['mixup'],
        erasing=aug_config['erasing'],
        
        project='exports',
        name='stack_detector',
        exist_ok=True
    )
    
    print("Training completed. Exporting models...")
    # Export model to ONNX for inference deployment if desired
    # model.export(format='onnx')

if __name__ == '__main__':
    main()
