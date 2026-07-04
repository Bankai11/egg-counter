import os
import cv2
import base64
import glob
import time
import threading
import argparse
import sys
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from predict import predict_trays, load_model

app = FastAPI(
    title="Egg Tray Detection API",
    description="API for detecting and counting egg trays in images, folders, and live cameras",
    version="1.0.0"
)

# Global model cache path
WEIGHTS_PATH = os.environ.get("EGG_TRAY_WEIGHTS", "egg_tray_training/yolov8_egg_tray/weights/best.pt")
if not os.path.exists(WEIGHTS_PATH) and os.path.exists("yolov8n.pt"):
    WEIGHTS_PATH = "yolov8n.pt"

print(f"FastAPI will use weights path: {WEIGHTS_PATH}")

# Request model for folder prediction
class FolderRequest(BaseModel):
    folder_path: str
    conf: float = 0.25
    iou: float = 0.45

class StreamStartRequest(BaseModel):
    source: str
    conf: float = 0.25
    iou: float = 0.45

# Create static directory if not exists
os.makedirs("static", exist_ok=True)

# Thread-safe Camera Streamer Class
class CameraStreamer:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.source = None
        self.conf = 0.25
        self.iou = 0.45
        self.lock = threading.Lock()
        self.latest_annotated_frame = None
        self.latest_metadata = {"count": 0, "fps": 0.0, "boxes": []}
        self.thread = None

    def start(self, source, conf=0.25, iou=0.45):
        self.stop()
        self.source = source
        self.conf = conf
        self.iou = iou
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"Camera Streamer thread launched for source: {source}")

    def _capture_loop(self):
        # Cast to integer if source is a pure digit (webcam index)
        src = int(self.source) if str(self.source).isdigit() else self.source
        self.cap = cv2.VideoCapture(src)
        
        # Optimize resolution for low-latency live inference
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        model = load_model(WEIGHTS_PATH)
        last_time = time.time()
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            # Perform inference on the frame
            start_infer = time.time()
            count, boxes, annotated = predict_trays(
                frame, 
                model, 
                conf_threshold=self.conf, 
                iou_threshold=self.iou
            )
            
            now = time.time()
            fps = 1.0 / (now - last_time) if (now - last_time) > 0 else 0.0
            last_time = now
            
            # Thread-safe updates of latest cached frame and statistics
            with self.lock:
                self.latest_annotated_frame = annotated
                self.latest_metadata = {
                    "count": count,
                    "fps": round(fps, 1),
                    "boxes": boxes
                }
                
        if self.cap:
            self.cap.release()
            self.cap = None

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.5)
            self.thread = None
        print("Camera Streamer thread terminated")

streamer = CameraStreamer()

# Module-level parsing of arguments to auto-start stream if needed
sys_args = sys.argv
default_source = None
default_conf = 0.25
default_iou = 0.45

for idx, val in enumerate(sys_args):
    if val == "--source" and idx + 1 < len(sys_args):
        default_source = sys_args[idx + 1]
    elif val == "--conf" and idx + 1 < len(sys_args):
        default_conf = float(sys_args[idx + 1])
    elif val == "--iou" and idx + 1 < len(sys_args):
        default_iou = float(sys_args[idx + 1])

@app.on_event("startup")
async def startup_event():
    if default_source is not None:
        print(f"Auto-starting camera stream from command-line: {default_source}")
        streamer.start(default_source, default_conf, default_iou)

@app.on_event("shutdown")
async def shutdown_event():
    streamer.stop()

@app.post("/predict-image")
async def api_predict_image(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file format")
            
        count, boxes, annotated_img = predict_trays(
            img,
            WEIGHTS_PATH,
            conf_threshold=conf,
            iou_threshold=iou
        )
        
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse(content={
            "detected_trays": count,
            "boxes": boxes,
            "annotated_image": f"data:image/jpeg;base64,{img_base64}"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/predict-folder")
async def api_predict_folder(req: FolderRequest):
    folder_path = req.folder_path
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail=f"Folder path '{folder_path}' does not exist.")
        
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
    image_paths = sorted(list(set(image_paths)))
    
    if not image_paths:
        return JSONResponse(content={
            "folder_path": folder_path,
            "total_images": 0,
            "processed_images": 0,
            "total_trays": 0,
            "results": {}
        })
        
    output_dir = os.path.join(folder_path, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    total_trays = 0
    processed_count = 0
    
    try:
        model = load_model(WEIGHTS_PATH)
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            try:
                count, boxes, annotated_img = predict_trays(
                    img_path,
                    model,
                    conf_threshold=req.conf,
                    iou_threshold=req.iou
                )
                
                out_path = os.path.join(output_dir, f"annotated_{filename}")
                cv2.imwrite(out_path, annotated_img)
                
                img_preview = None
                if processed_count < 10:
                    _, buffer = cv2.imencode('.jpg', annotated_img)
                    img_preview = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                
                results[filename] = {
                    "count": count,
                    "boxes": boxes,
                    "annotated_path": os.path.abspath(out_path),
                    "preview": img_preview
                }
                total_trays += count
                processed_count += 1
            except Exception as inner_e:
                print(f"Error on image {filename}: {inner_e}")
                
        return JSONResponse(content={
            "folder_path": os.path.abspath(folder_path),
            "total_images": len(image_paths),
            "processed_images": processed_count,
            "total_trays": total_trays,
            "output_directory": os.path.abspath(output_dir),
            "results": results
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch folder processing failed: {str(e)}")

# Real-Time Stream Endpoint (MJPEG)
@app.get("/video-feed")
async def video_feed():
    """
    Returns an MJPEG video streaming boundary frame sequence.
    """
    if not streamer.is_running:
        raise HTTPException(status_code=400, detail="Camera capture stream is not running. Please start it first.")
        
    def frame_generator():
        while streamer.is_running:
            frame = None
            with streamer.lock:
                if streamer.latest_annotated_frame is not None:
                    frame = streamer.latest_annotated_frame.copy()
            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.04) # Cap frame push rate at ~25 FPS to optimize bandwidth
            
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/start-stream")
async def start_stream(req: StreamStartRequest):
    try:
        streamer.start(req.source, req.conf, req.iou)
        return {"status": "success", "message": f"Stream started on source: {req.source}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop-stream")
async def stop_stream():
    try:
        streamer.stop()
        return {"status": "success", "message": "Stream stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream-status")
async def get_stream_status():
    with streamer.lock:
        metadata = streamer.latest_metadata.copy()
    return {
        "is_running": streamer.is_running,
        "source": streamer.source,
        "tray_count": metadata.get("count", 0),
        "fps": metadata.get("fps", 0.0),
        "boxes": metadata.get("boxes", [])
    }

@app.get("/status")
async def get_status():
    import torch
    model_loaded = os.path.exists(WEIGHTS_PATH)
    return {
        "status": "online",
        "weights_path": os.path.abspath(WEIGHTS_PATH),
        "weights_found": model_loaded,
        "cuda_available": torch.cuda.is_available() if torch else False,
        "device_name": torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else "CPU"
    }

# Serve Frontend statically
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return f.read()
    return """
    <html>
        <head><title>Egg Tray Counter</title></head>
        <body style="font-family: sans-serif; padding: 50px; background: #121212; color: #fff;">
            <h2>Egg Tray Counting System Backend</h2>
            <p>Dashboard HTML not found in static folder.</p>
        </body>
    </html>
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Start Egg Tray Counter API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--source", type=str, default=None, help="Camera index or RTSP source to auto-start")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU NMS threshold")
    
    args, unknown = parser.parse_known_args()
    
    # If starting via CLI with a live camera source, disable reload to prevent double thread instantiation
    should_reload = True if args.source is None else False
    
    uvicorn.run("app:app", host=args.host, port=args.port, reload=should_reload)

