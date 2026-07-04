# 🥚 Egg Tray Counter

Welcome to the **Egg Tray Counter**! 🐔 This project uses a custom-trained YOLO object detection model to instantly detect and count egg trays in images, video files, or a live webcam feed. 

It comes with a beautiful **Web Dashboard** powered by FastAPI, making it super easy to use right from your browser!

---

## 🚀 Features

- **Web Dashboard:** A clean, easy-to-use browser interface to upload images, process videos, or stream your camera.
- **Live Webcam Counting:** Point your camera at egg trays and count them in real-time.
- **Image & Video Support:** Process saved photos and MP4 files effortlessly.
- **Highly Accurate:** Powered by a custom-trained YOLO model optimized specifically for egg trays.

---

## 🛠️ Step-by-Step Installation

Follow these simple steps to get the project running on your computer.

### 1. Open the Project Folder
First, open your terminal (Command Prompt or PowerShell) and navigate to the project folder. For example:
```powershell
cd C:\Users\banka\Documents\egg-tray-counter
```

### 2. Activate the Virtual Environment
We use a virtual environment (`venv`) to keep all the project dependencies cleanly separated from your system. Activate it with this command:
```powershell
.\venv\Scripts\activate
```
*(You should see `(venv)` appear at the beginning of your terminal line).*

### 3. Install Dependencies
Make sure you have all the required libraries installed by running:
```powershell
pip install -r requirements.txt
```

---

## 🎮 How to Use

There are two ways to use the Egg Tray Counter: the **Web Dashboard** (Recommended) or the **Command Line**.

### Option A: The Web Dashboard (Easiest!) 🌐

We built a beautiful web interface so you don't have to deal with terminal commands every time.

1. Start the web server:
   ```powershell
   python app.py
   ```
2. Open your web browser and go to: **[http://localhost:8000](http://localhost:8000)**
3. From the dashboard, you can effortlessly upload images, process video files, or start your live webcam feed!

### Option B: Command Line Interface (CLI) 💻

If you prefer using the terminal directly, you can use the built-in script.

**1. Live Webcam Feed**
```powershell
python scripts/inference.py --source 0
```
*(A window will pop up. Click on it and press **`q`** to stop the video feed)*

**2. Count a Saved Video**
```powershell
python scripts/inference.py --source "path/to/video.mp4" --output "counted_video.mp4"
```

**3. Count a Static Image**
```powershell
python scripts/inference.py --source "path/to/image.jpg" --output "counted_image.jpg"
```

---

## ⚙️ Pro-Tips

**Adjusting the Confidence:** 
If the model is detecting things that aren't egg trays (false positives), or if it's missing some trays, you can adjust the *confidence threshold*. The default is `0.25` (25%). 

To make it stricter, increase the number:
```powershell
python scripts/inference.py --source 0 --conf 0.50
```

Happy Counting! 🎉
