import cv2
import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import supervision as sv
import argparse
import time
from datetime import datetime

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--record', type=int, help='Start recording after N seconds')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model = YOLO(model_path)
model.eval()

box_annotator = sv.BoxAnnotator()

smoothed_cx, smoothed_cy = None, None
alpha = 0.03

zoom_factor = 1.0 
zoom_in_animation = 0.05

# Recording setup
video_writer = None
start_time = time.time()
recording_started = False

cap = cv2.VideoCapture(0)  # Open webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip image horizontly
    frame = cv2.flip(frame, 1)

    # Inference
    with torch.no_grad():
        output = model(frame, verbose=False)
        predictions = Detections.from_ultralytics(output[0])

    # Zoom and Center
    h, w = frame.shape[:2]

    # Zoom-in Animation
    if zoom_factor < 1.5: # Adjust this value to control zoom level
        zoom_factor += zoom_in_animation

    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)

    if len(predictions) > 0:
        # Get the first detection's bounding box
        x1, y1, x2, y2 = predictions.xyxy[0]
        target_cx = (x1 + x2) / 2
        target_cy = (y1 + y2) / 2
    else:
        target_cx = w / 2
        target_cy = h / 2

    if smoothed_cx is None:
        smoothed_cx = target_cx
        smoothed_cy = target_cy
    else:
        smoothed_cx = smoothed_cx * (1 - alpha) + target_cx * alpha
        smoothed_cy = smoothed_cy * (1 - alpha) + target_cy * alpha

    # Calculate top-left corner of the crop area
    x = smoothed_cx - new_w / 2
    y = smoothed_cy - new_h / 2

    # Clamp to ensure we don't go out of bounds (avoid borders)
    x = max(0, min(x, w - new_w))
    y = max(0, min(y, h - new_h))

    # Handle Recording
    if args.record is not None:
        elapsed_time = time.time() - start_time
        if elapsed_time >= args.record:
            if not recording_started:
                print("Recording started...")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                filename = f"center-frame-{timestamp}.mp4"
                video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
                recording_started = True
            
            # Create clean zoomed frame for recording (no annotations)
            clean_crop = frame[int(y):int(y + new_h), int(x):int(x + new_w)]
            clean_zoomed = cv2.resize(clean_crop, (w, h))
            video_writer.write(clean_zoomed)

    # Visualization (Annotate AFTER recording clean frame)
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=predictions
    )

    # Crop and resize for display (with annotations)
    crop = annotated_frame[int(y):int(y + new_h), int(x):int(x + new_w)]
    annotated_frame = cv2.resize(crop, (w, h))

    # UI Overlays
    if args.record is not None:
        elapsed_time = time.time() - start_time
        if elapsed_time < args.record:
            remaining = int(args.record - elapsed_time) + 1
            text = f"Recording in {remaining}s"
            cv2.putText(annotated_frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif recording_started:
             cv2.circle(annotated_frame, (30, 30), 10, (0, 0, 255), -1)  # Red dot

    cv2.imshow("YOLOv8 Face Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if video_writer:
    video_writer.release()

cap.release()
cv2.destroyAllWindows()