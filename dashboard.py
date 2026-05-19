import streamlit as st
import cv2
import torch
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from datetime import datetime
import os

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Zero-Shot Surveillance System",
    layout="wide"
)

st.title("AI Zero-Shot Surveillance Dashboard")

st.markdown("""
### System Features

- YOLOv8 Object Detection
- CLIP Zero-Shot Classification
- Suspicious Activity Detection
- Automated Alerts
- Real-Time Monitoring
""")

# ---------------------------------------------------
# CREATE ALERT FOLDER
# ---------------------------------------------------
os.makedirs("outputs/alerts", exist_ok=True)

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------
@st.cache_resource
def load_models():

    # YOLO model
    yolo_model = YOLO("yolov8n.pt")

    # CLIP model
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    # CLIP processor
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    return yolo_model, clip_model, processor


yolo_model, clip_model, processor = load_models()

# ---------------------------------------------------
# DEVICE SETUP
# ---------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model.to(device)

st.sidebar.success(f"Using device: {device}")

# ---------------------------------------------------
# ZERO-SHOT LABELS
# ---------------------------------------------------
labels = [
    "person standing",
    "person walking",
    "person near equipment",
    "person inspecting machine",
    "normal activity",
    "suspicious activity"
]

# Labels considered suspicious
alert_labels = [
    "suspicious activity"
]

# ---------------------------------------------------
# DASHBOARD SIDEBAR
# ---------------------------------------------------
alert_count = 0

alert_box = st.sidebar.empty()

frame_placeholder = st.empty()

# ---------------------------------------------------
# VIDEO SOURCE
# ---------------------------------------------------
video_path = "videos/sample.mp4"

cap = cv2.VideoCapture(video_path)

# Check video
if not cap.isOpened():
    st.error("Could not open video.")
    st.stop()

# ---------------------------------------------------
# FRAME COUNTER
# ---------------------------------------------------
frame_count = 0

# ---------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ---------------------------------------------------
    # FRAME SKIPPING FOR PERFORMANCE
    # ---------------------------------------------------
    frame_count += 1

    if frame_count % 5 != 0:
        continue

    # ---------------------------------------------------
    # RESIZE FRAME
    # ---------------------------------------------------
    frame = cv2.resize(frame, (900, 500))

    # ---------------------------------------------------
    # YOLO DETECTION
    # ---------------------------------------------------
    results = yolo_model(frame)

    for result in results:

        boxes = result.boxes

        for i, box in enumerate(boxes):

            # Process only first 2 detections
            if i >= 2:
                break

            # ---------------------------------------------------
            # PERSON ONLY FILTER
            # ---------------------------------------------------
            class_id = int(box.cls[0])

            # YOLO class 0 = person
            if class_id != 0:
                continue

            # ---------------------------------------------------
            # BOUNDING BOX
            # ---------------------------------------------------
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = frame[y1:y2, x1:x2]

            # Skip invalid crops
            if crop.size == 0:
                continue

            # ---------------------------------------------------
            # CONVERT IMAGE
            # ---------------------------------------------------
            image = Image.fromarray(
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            )

            # ---------------------------------------------------
            # CLIP PROCESSING
            # ---------------------------------------------------
            inputs = processor(
                text=labels,
                images=image,
                return_tensors="pt",
                padding=True
            )

            # Move to device
            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }

            # ---------------------------------------------------
            # CLIP INFERENCE
            # ---------------------------------------------------
            with torch.no_grad():

                outputs = clip_model(**inputs)

            logits_per_image = outputs.logits_per_image

            probs = logits_per_image.softmax(dim=1)

            # Best prediction
            best_idx = probs.argmax().item()

            best_label = labels[best_idx]

            confidence = probs[0][best_idx].item()

            # ---------------------------------------------------
            # CONFIDENCE FILTER
            # ---------------------------------------------------
            if confidence < 0.50:
                continue

            # ---------------------------------------------------
            # DRAW BOX
            # ---------------------------------------------------
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ---------------------------------------------------
            # LABEL TEXT
            # ---------------------------------------------------
            text = f"{best_label}: {confidence:.2f}"

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # ---------------------------------------------------
            # ALERT SYSTEM
            # ---------------------------------------------------
            if best_label in alert_labels:

                alert_count += 1

                alert_box.error(
                    f"Alerts Detected: {alert_count}"
                )

                # Timestamp
                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                # Save screenshot
                filename = (
                    f"outputs/alerts/"
                    f"{best_label}_{timestamp}.jpg"
                )

                cv2.imwrite(filename, frame)

    # ---------------------------------------------------
    # DISPLAY FRAME
    # ---------------------------------------------------
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    frame_placeholder.image(
        frame_rgb,
        channels="RGB",
        use_container_width=True
    )

# ---------------------------------------------------
# CLEANUP
# ---------------------------------------------------
cap.release()

st.success("Surveillance processing completed.")