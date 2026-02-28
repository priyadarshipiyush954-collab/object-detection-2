"""
Run YOLO object detection with webcam input.
Press 'q' to quit.
"""

from __future__ import annotations
import cv2
import cvzone
from ultralytics import YOLO


def main():

    # Load YOLO model
    model = YOLO("yolo-weights/yolov8n.pt")
    class_names = model.names

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        # 🔥 Flip camera horizontally (mirror mode)
        frame = cv2.flip(frame, 1)

        # Run YOLO inference
        results = model(frame, conf=0.4, imgsz=640, verbose=False)[0]

        # Process detections
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = f"{class_names[cls_id]} {confidence:.2f}"

            # Draw stylish rectangle
            cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2)

            # Draw label
            cvzone.putTextRect(
                frame,
                label,
                (max(0, x1), max(35, y1)),
                scale=1,
                thickness=1
            )

        cv2.imshow("YOLO Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()