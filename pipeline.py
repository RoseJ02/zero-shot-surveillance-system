import cv2
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime
from PIL import Image
import torch


print("Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")

print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)
# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

clip_model.to(device)

print("Opening video...")

# VIDEO PATH
video_path = "videos/sample.mp4"

cap = cv2.VideoCapture(video_path)

# Check video
if not cap.isOpened():
    print("ERROR: Could not open video.")
    print("Check your video path.")
    exit()

print("Video opened successfully!")

labels = [
    "person",
    "car",
    "bicycle",
    "normal activity",
    "suspicious activity"
]
alert_labels = [
    "suspicious activity",
    "person fighting",
    "person with weapon",
    "person stealing"
]
frame_count = 0

while True:

    ret, frame = cap.read()
    frame_count += 1

    # Process every 5th frame
    if frame_count % 5 != 0:
        continue

    if not ret:
        print("Video ended or failed.")
        break

    frame = cv2.resize(frame, (640, 360))

    results = yolo_model(frame)

    for result in results:

        boxes = result.boxes

        # Process only ONE object
        for i, box in enumerate(boxes):

            if i >= 1:
                break

            # Get YOLO class ID
            class_id = int(box.cls[0])

            # Process only persons
            # YOLO class 0 = person
            if class_id != 0:
                continue

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            image = Image.fromarray(
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            )

            inputs = processor(
                text=labels,
                images=image,
                return_tensors="pt",
                padding=True
            )
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = clip_model(**inputs)

            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

            best_idx = probs.argmax().item()
            best_label = labels[best_idx]

            confidence = probs[0][best_idx].item()
            # Ignore weak predictions
            # Ignore weak predictions
            if confidence < 0.30:
                continue

            # ALERT SYSTEM
            if best_label in alert_labels:

                # Timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Screenshot filename
                filename = f"outputs/alerts/{best_label}_{timestamp}.jpg"

                # Save screenshot
                cv2.imwrite(filename, frame)

                print(f"ALERT: {best_label} detected!")
                print(f"Saved: {filename}")

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

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

    cv2.imshow("Zero-Shot Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()