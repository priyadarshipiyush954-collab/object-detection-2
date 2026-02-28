"""Run YOLO object detection with webcam input.

Press `q` to quit.
"""

from __future__ import annotations

import argparse
from pathlib import Path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 webcam detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("yolo-weights/yolov8n.pt"),
        help="Path to YOLO weights file",
    )
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Mirror webcam feed horizontally",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import cv2
    import cvzone
    from ultralytics import YOLO

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera}")

    model = YOLO(args.weights)
    class_names = model.names

    while True:
        success, frame = cap.read()
        if not success:
            print("Warning: unable to read frame from camera.")
            break

        if args.flip:
            frame = cv2.flip(frame, 1)

        result = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            width, height = x2 - x1, y2 - y1

            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = f"{class_names[cls_id]} {confidence:.2f}"

            cvzone.cornerRect(frame, (x1, y1, width, height), l=9, rt=2)
            cvzone.putTextRect(frame, label, (max(0, x1), max(35, y1)), scale=1, thickness=1)

        cv2.imshow("YOLO Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
