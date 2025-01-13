# Put this file in the yolov9 folder
import cv2
import torch
import time
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

# Khởi tạo YOLOv9
device = "cuda"
device = torch.device(device)
model  = DetectMultiBackend(weights="weights/yolov9-s-converted.pt", device=device, fuse=True )
model  = AutoShape(model)

# Load classname từ file classes.names
with open("data_ext/classes.names") as f:
    class_names = f.read().strip().split('\n')
print(class_names)
colors = np.random.randint(0,255, size=(len(class_names),3 ))

# Initialize VideoCapture to read from a video source
cap = cv2.VideoCapture(1)

# Initialize DeepSort
tracker = DeepSort(max_age=30)
tracks = []

# Variables to calculate FPS
prev_time = 0

# Process each frame
while True:
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        break

    # Get the current time
    current_time = time.time()
    
    # Calculate FPS
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    # Pass the frame through the model for detection (replace `model` with your actual model)
    results = model(frame)
    detect = []
    for detect_object in results.pred[0]:
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)
        
        if confidence < conf_threshold:
            continue
        detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    # Update tracks using DeepSort
    tracks = tracker.update_tracks(detect, frame=frame)

    # Draw bounding boxes and labels
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = (255, 0, 0)  # Set your color logic
            label = f"ID-{track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

