import os
import sys
import torch
import cv2
import numpy as np
import easyocr
import pathlib

# Fix Pathlib compatibility
pathlib.PosixPath = pathlib.WindowsPath

# --- Setup base paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                 # app1/
YOLOV5_DIR = os.path.join(BASE_DIR, '..', 'yolov5')                  # ../yolov5
MODEL_PATH = os.path.join(YOLOV5_DIR, 'best.pt')                     # ../yolov5/best.pt

# Add YOLOv5 to system path
sys.path.append(YOLOV5_DIR)

# --- YOLOv5 Imports ---
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.general import LOGGER
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox

# --- Device ---
device = select_device('0' if torch.cuda.is_available() else 'cpu')

# --- Load Model ---
print("DEBUG — Model Path Exists:", os.path.exists(MODEL_PATH))  # ✅ DEBUG 1
model = DetectMultiBackend(MODEL_PATH, device=device)
model.eval()
model.warmup(imgsz=(1, 3, 640, 640))
print("✅ YOLOv5 model loaded and ready.")

# --- Load OCR Reader ---
reader = easyocr.Reader(['en'])

# --- Main Detection Function ---
def run_detection(img_path):
    print("DEBUG — Image Path:", img_path)  # ✅ DEBUG 2
    img0 = cv2.imread(img_path)
    print("DEBUG — Image Read:", img0.shape if img0 is not None else "Failed to read")  # ✅ DEBUG 3
    if img0 is None:
        print(f"❌ Error: Cannot read image from {img_path}")
        return {"detections": [], "ocr_results": []}

    # Prepare image
    img = letterbox(img0, new_shape=640, stride=32)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)

    # Predict
    pred = model(img, augment=False, visualize=False)
    print("DEBUG — Raw Predictions:", pred)  # ✅ DEBUG 4
    pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.45)
    print("DEBUG — Filtered Predictions:", pred)  # ✅ DEBUG 5

    detections = []
    ocr_results = []
    class_names = model.names

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                class_id = int(cls)
                label = class_names[class_id]
                confidence = float(conf)

                print(f"DEBUG — Detection Found: {label}, ID: {class_id}, Conf: {confidence:.2f}")  # ✅ DEBUG 6

                # Save detection
                detections.append({
                    "class": label,
                    "class_id": class_id,
                    "conf": confidence,
                    "box": [x1, y1, x2, y2]
                })

                # Draw box on image (optional)
                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img0, f"{label} {confidence:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Run OCR if number plate detected
                if label.lower() in ["number plate", "plate"]:  # Adjust for your class name
                    cropped = img0[y1:y2, x1:x2]
                    print(f"DEBUG — Running OCR on plate region: {cropped.shape}")  # ✅ DEBUG 7

                    # Preprocess for OCR
                    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    _, threshold_cropped = cv2.threshold(gray_cropped, 127, 255, cv2.THRESH_BINARY)

                    # OCR processing
                    ocr = reader.readtext(threshold_cropped)
                    for (_, text, ocr_conf) in ocr:
                        print(f"DEBUG — OCR Found: {text} with conf {ocr_conf:.2f}")  # ✅ DEBUG 8
                        ocr_results.append({
                            "text": text,
                            "conf": float(ocr_conf)
                        })

    return {
        "detections": detections,
        "ocr_results": ocr_results
    }
