# Object Detection (YOLOv8)

Simple Python scripts for object detection with **Ultralytics YOLOv8** on:
- a single image (`yolo.py`)
- a live webcam stream (`yolo_web.py`)

## Features
- Configurable model weights
- Confidence threshold control for cleaner detections
- Adjustable inference image size (`imgsz`) for speed/accuracy trade-off
- Optional output save for image detections

## Project Structure
- `yolo.py` - Detect objects on a static image and display the result.
- `yolo-weights/chapter 6 - yolo with webcam/yolo_web.py` - Real-time webcam detection.
- `yolo-weights/yolov8n.pt` - YOLOv8 nano weights.
- `chapter 5 - yolo/School Bus On Road.png` - Sample image.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1) Image detection
```bash
python yolo.py \
  --image "chapter 5 - yolo/School Bus On Road.png" \
  --weights "yolo-weights/yolov8n.pt" \
  --conf 0.35 \
  --imgsz 640 \
  --save outputs/detected_image.jpg
```

### 2) Webcam detection
```bash
python "yolo-weights/chapter 6 - yolo with webcam/yolo_web.py" \
  --weights "yolo-weights/yolov8n.pt" \
  --conf 0.40 \
  --imgsz 640 \
  --flip
```
Press `q` to exit.

## Tips for better accuracy
- Use a larger model (for example `yolov8m.pt` / `yolov8l.pt`) if your hardware allows it.
- Increase `imgsz` (e.g., 960) for small or distant objects.
- Tune `--conf`:
  - lower values detect more objects but may add false positives
  - higher values reduce false positives but can miss weak detections
- Ensure good lighting and stable camera framing for webcam detection.

## License
This project is licensed under the MIT License. See `LICENSE`.
