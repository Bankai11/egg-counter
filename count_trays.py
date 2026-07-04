import os
import glob
import cv2
import time
import argparse
import json
from predict import predict_trays, load_model

def process_image(image_path, model, conf_threshold, iou_threshold, output_dir):
    try:
        count, boxes, annotated_img = predict_trays(
            image_path,
            model,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        # Save annotated image
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"annotated_{filename}")
            cv2.imwrite(output_path, annotated_img)
            
        return count, boxes, True
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return 0, [], False

def process_folder(folder_path, model, conf_threshold, iou_threshold, output_dir):
    # Support multiple formats
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_paths = []
    for ext in extensions:
        # Case insensitive patterns
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
    image_paths = sorted(list(set(image_paths)))
    
    if not image_paths:
        print(f"No images found in folder: {folder_path}")
        return
        
    print(f"Found {len(image_paths)} images in {folder_path}. Processing...")
    
    results = {}
    total_trays = 0
    success_count = 0
    
    start_time = time.time()
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        count, boxes, success = process_image(img_path, model, conf_threshold, iou_threshold, output_dir)
        if success:
            results[filename] = {
                "count": count,
                "boxes": boxes
            }
            total_trays += count
            success_count += 1
            print(f"Image: {filename} | Detected trays: {count} | Boxes: {boxes}")
            
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*30)
    print("=== Folder Processing Summary ===")
    print("="*30)
    print(f"Folder: {os.path.abspath(folder_path)}")
    print(f"Total Images Found: {len(image_paths)}")
    print(f"Successfully Processed: {success_count}")
    print(f"Total Trays Detected: {total_trays}")
    if success_count > 0:
        print(f"Average Trays per Image: {total_trays / success_count:.2f}")
    print(f"Total Time Elapsed: {elapsed_time:.2f} seconds")
    print(f"Annotated outputs saved in: {os.path.abspath(output_dir) if output_dir else 'None'}")
    print("="*30)
    
    # Save a summary JSON
    if output_dir:
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump({
                "folder": os.path.abspath(folder_path),
                "total_images": len(image_paths),
                "processed_images": success_count,
                "total_trays": total_trays,
                "detections": results
            }, f, indent=4)
        print(f"Summary JSON saved to: {summary_path}")

def process_video(video_path, model, conf_threshold, iou_threshold, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing Video: {video_path}")
    print(f"Resolution: {width}x{height} | FPS: {fps} | Total Frames: {total_frames}")
    
    out_writer = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_name = f"annotated_{os.path.basename(video_path)}"
        out_path = os.path.join(output_dir, out_name)
        # Use MP4V codec which is highly compatible
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        # Process every frame
        count, boxes, annotated_frame = predict_trays(
            frame,
            model,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        # Display overlay tracking progress in console
        if frame_count % 30 == 0 or frame_count == total_frames:
            print(f"Frame {frame_count}/{total_frames} | Detected trays: {count}")
            
        if out_writer:
            out_writer.write(annotated_frame)
            
    cap.release()
    if out_writer:
        out_writer.release()
        
    elapsed = time.time() - start_time
    print(f"Video processing finished. Total frames: {frame_count} in {elapsed:.2f}s (~{frame_count/elapsed:.1f} FPS)")
    if output_dir:
        print(f"Annotated video saved to: {os.path.abspath(out_path)}")

def main():
    parser = argparse.ArgumentParser(description="Count egg trays from images or videos using YOLOv8")
    parser.add_argument("--source", type=str, required=True, help="Path to image file, folder of images, or video file")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to YOLOv8 weights file (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save annotated output images/videos")
    
    args = parser.parse_args()
    
    # Load model once for speed
    print(f"Loading weights {args.weights}...")
    model = load_model(args.weights)
    
    source_path = args.source
    
    if os.path.isdir(source_path):
        process_folder(source_path, model, args.conf, args.iou, args.output_dir)
    elif os.path.isfile(source_path):
        # Check if it is a video or image
        ext = os.path.splitext(source_path)[1].lower()
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.mpeg']
        
        if ext in video_exts:
            process_video(source_path, model, args.conf, args.iou, args.output_dir)
        else:
            # Assume image
            print(f"Processing single image: {source_path}")
            count, boxes, success = process_image(source_path, model, args.conf, args.iou, args.output_dir)
            if success:
                print(f"Detected trays: {count}")
                print(f"Boxes: {boxes}")
    else:
        print(f"Error: Source path '{source_path}' does not exist.")

if __name__ == '__main__':
    main()
