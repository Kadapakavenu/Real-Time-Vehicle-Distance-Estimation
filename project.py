import cv2
import torch
import numpy as np
from ultralytics import YOLO
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  # Use yolov8n.pt for speed or yolov8m.pt for better accuracy

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to("cuda" if torch.cuda.is_available() else "cpu").eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for MiDaS
    input_midas = midas_transforms(frame).to("cuda" if torch.cuda.is_available() else "cpu")

    # Predict depth
    with torch.no_grad():
        prediction = midas(input_midas)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Object detection
    results = yolo_model(frame)[0]
    for box in results.boxes:
        cls = int(box.cls[0])
        if yolo_model.names[cls] == "car":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = depth_map[y1:y2, x1:x2]
            if bbox.size > 0:
                avg_depth = np.mean(bbox)
                label = f"Car: {avg_depth:.2f}m"
                color = (0, 0, 255) if avg_depth < 10 else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Vehicle Distance Estimation", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()