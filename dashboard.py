import streamlit as st
import cv2
import torch
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from datetime import datetime

# -----------------------------------
# Page Config
# -----------------------------------
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

# -----------------------------------
# Load Models
# -----------------------------------
@st.cache_resource
def load_models():

    yolo_model = YOLO("yolov8n.pt")

    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    return yolo_model, clip_model, processor

yolo_model, clip_model, processor = load_models()

# -----------------------------------
# Device Setup
# -----------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model.to(device)

st.sidebar.success(f"Using device: {device}")

# -----------------------------------
# Labels
# -----------------------------------
labels = [
    "normal activity",
    "person walking",
    "person running",
    "person fighting",
    "person stealing",
    "person with backpack",
    "person with weapon",
    "suspicious activity"
]

alert_labels = [
    "person fighting",
    "person stealing",
    "person with weapon",
    "suspicious activity"
]

# -----------------------------------
# Dashboard Metrics
# -----------------------------------
alert_count = 0

alert_box = st.sidebar.empty()

frame_placeholder = st.empty()

# -----------------------------------
# Open Video
# -----------------------------------
cap = cv2.VideoCapture("videos/sample.mp4")

frame_count = 0

# -----------------------------------
# Main Loop
# -----------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Process every 5th frame
    if frame_count % 5 != 0:
        continue

    # Resize frame
    frame = cv2.resize(frame, (800, 450))

    # YOLO detection
    results = yolo_model(frame)

    for result in results:

        boxes = result.boxes

        for i, box in enumerate(boxes):

            # Process only first 2 detections
            if i >= 2:
                break

            # Person only
            class_id = int(box.cls[0])

            if class_id != 0:
                continue

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            image = Image.fromarray(
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            )

            # CLIP processing
            inputs = processor(
                text=labels,
                images=image,
                return_tensors="pt",
                padding=True
            )

            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }

            # CLIP inference
            with torch.no_grad():

                outputs = clip_model(**inputs)

            logits_per_image = outputs.logits_per_image

            probs = logits_per_image.softmax(dim=1)

            best_idx = probs.argmax().item()

            best_label = labels[best_idx]

            confidence = probs[0][best_idx].item()

            # Ignore weak predictions
            if confidence < 0.30:
                continue

            # Draw box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw label
            text = f"{best_label}: {confidence:.2f}"

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Alerts
            if best_label in alert_labels:

                alert_count += 1

                alert_box.error(
                    f"Alerts Detected: {alert_count}"
                )

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                filename = (
                    f"outputs/alerts/"
                    f"{best_label}_{timestamp}.jpg"
                )

                cv2.imwrite(filename, frame)

    # Convert BGR → RGB
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Display frame
    frame_placeholder.image(
        frame_rgb,
        channels="RGB",
        use_container_width=True
    )

cap.release()

st.success("Surveillance processing completed.")