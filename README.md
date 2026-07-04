# Egg Tray Counter 🥚

This project uses a custom-trained YOLO11 object detection model to count the number of egg trays in an image, a video file, or from a live webcam feed.

## Prerequisites

Whenever you want to run this code, make sure you are in the project folder and you have activated the virtual environment:
```powershell
cd C:\Users\banka\Documents\egg-tray-counter
.\venv\Scripts\activate
```

## How to Run

Use the `scripts/inference.py` file to run the model.

### 1. Run on a Live Webcam (Recommended)
To start a live stream using your computer's default webcam (usually ID `0`):
```powershell
python scripts/inference.py --source 0
```
- A window will pop up showing the live camera feed with boxes and counts!
- It will automatically save the recording to `output.mp4`.
- **To Stop:** Click on the video window and press the **`q`** key on your keyboard.

### 2. Run on a Saved Video File
If you have an MP4 file you want to count:
```powershell
python scripts/inference.py --source "path/to/video.mp4" --output "counted_video.mp4"
```
You can watch it live as it processes, and the final result will be saved to `counted_video.mp4`.

### 3. Run on a Single Image
If you just want to count a static picture:
```powershell
python scripts/inference.py --source "path/to/image.jpg" --output "counted_image.jpg"
```

### Options
You can adjust the confidence threshold if the model is predicting too many false positives, or missing trays. The default is `0.25` (25% confidence). To make it stricter, pass a higher number:
```powershell
python scripts/inference.py --source 0 --conf 0.50
```
