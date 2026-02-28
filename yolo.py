from ultralytics import YOLO
import cv2

model = YOLO("../yolo-weights/yolov8l.pt")

results = model(r"D:\object detection 2\chapter 5 - yolo\School Bus On Road.png")
annotated_frame = results[0].plot()

cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
cv2.imshow("YOLO Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()