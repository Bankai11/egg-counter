import argparse
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

def process_image(model, img_path, out_path, conf_thresh=0.25):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image: {img_path}")
        return

    results = model.predict(img, conf=conf_thresh, verbose=False)
    
    # Get total count
    count = len(results[0].boxes)
    
    # Plot results on image
    annotated_img = results[0].plot()
    
    # Add count text
    text = f"Total Trays: {count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    
    # Text background
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(annotated_img, (10, 10), (10 + text_w + 20, 10 + text_h + 30), (0, 0, 0), -1)
    # Text
    cv2.putText(annotated_img, text, (20, 10 + text_h + 15), font, font_scale, (0, 255, 0), thickness)
    
    cv2.imwrite(out_path, annotated_img)
    print(f"Processed image. Count: {count}. Saved to {out_path}")

def process_video(model, vid_path, out_path, conf_thresh=0.25):
    # Check if vid_path is a webcam ID (integer)
    if vid_path.isdigit():
        vid_path = int(vid_path)
        is_webcam = True
    else:
        is_webcam = False

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"Error loading video or webcam: {vid_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0 # Fallback for webcams
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    source_name = "Webcam" if is_webcam else vid_path
    print(f"Processing {source_name}... Press 'q' to stop.")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.predict(frame, conf=conf_thresh, verbose=False)
        count = len(results[0].boxes)
        
        annotated_frame = results[0].plot()
        
        text = f"Total Trays: {count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(annotated_frame, (10, 10), (10 + text_w + 20, 10 + text_h + 30), (0, 0, 0), -1)
        cv2.putText(annotated_frame, text, (20, 10 + text_h + 15), font, font_scale, (0, 255, 0), thickness)
        
        out.write(annotated_frame)
        
        # Display the frame in a window
        cv2.imshow("Egg Tray Counter (Press 'q' to quit)", annotated_frame)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping early by user request...")
            break
            
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Finished processing. Saved recording to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Run YOLO11 inference on image, video, or webcam.")
    parser.add_argument("--source", type=str, required=True, help="Path to input image/video, or '0' for webcam.")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    args = parser.parse_args()

    weights_path = r"C:\Users\banka\runs\detect\runs\detect\yolo11m_egg_tray\weights\best.pt"
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return
        
    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)
    
    # Check if webcam (a digit like '0')
    if args.source.isdigit():
        process_video(model, args.source, args.output, args.conf)
        return
        
    ext = os.path.splitext(args.source)[1].lower()
    
    # Check if video or image
    video_exts = ['.mp4', '.avi', '.mov', '.mkv']
    if ext in video_exts:
        process_video(model, args.source, args.output, args.conf)
    else:
        # Default output to jpg if they didn't specify properly for an image
        if os.path.splitext(args.output)[1].lower() in video_exts:
            args.output = os.path.splitext(args.output)[0] + ".jpg"
        process_image(model, args.source, args.output, args.conf)

if __name__ == "__main__":
    main()
