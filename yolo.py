"""Run YOLO object detection on a single image.

Example:
    python yolo.py --image "chapter 5 - yolo/School Bus On Road.png"
"""

from __future__ import annotations

import argparse
from pathlib import Path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 object detection on an image")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("chapter 5 - yolo/School Bus On Road.png"),
        help="Path to input image",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("yolo-weights/yolov8n.pt"),
        help="Path to YOLO weights file",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold (0-1)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output path to save annotated image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    import cv2
    from ultralytics import YOLO

    model = YOLO(args.weights)
    results = model(args.image, conf=args.conf, imgsz=args.imgsz)
    annotated_frame = results[0].plot()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save), annotated_frame)
        print(f"Saved annotated image to: {args.save}")

    cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLO Detection", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
