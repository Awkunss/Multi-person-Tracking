
# YOLOv9 Object Tracking with DeepSort

This project integrates the YOLOv9 object detection model with the DeepSort tracker for real-time object tracking in videos or webcam feeds.

## Project Structure

- `weights/yolov9-s-converted.pt`: YOLOv9 model weights file.
- `data_ext/classes.names`: File containing the class names, one per line.
- `walking_1.mp4`: Example video file for testing.
- `main.py`: Python script containing the code for object detection and tracking.

## Usage

1. Clone this repository and navigate to the `yolov9` folder:
   ```bash
   git clone <repository-url>
   cd yolov9
   ```

2. Place the YOLOv9 weights file (`yolov9-s-converted.pt`) in the `weights` folder.

3. Create or verify the `data_ext/classes.names` file, which should list the class names used by your model.

4. Run the script:
   ```bash
   python main.py
   ```

5. Choose the input type when prompted:
   - `1` for a video file (default: `walking_1.mp4`).
   - `2` for live webcam input.

6. Press `q` to exit the program at any time.

## Code Highlights

### Initializing YOLOv9
The YOLOv9 model is initialized with:
```python
model = DetectMultiBackend(weights="weights/yolov9-s-converted.pt", device=device, fuse=True)
model = AutoShape(model)
```

### Object Detection
The YOLOv9 model processes each frame to detect objects:
```python
results = model(frame)
```

### Tracking with DeepSort
The DeepSort tracker maintains track IDs for detected objects:
```python
tracker = DeepSort(max_age=30)
tracks = tracker.update_tracks(detect, frame=frame)
```

### FPS Calculation
Frames per second (FPS) are calculated to monitor performance:
```python
fps = 1 / (current_time - prev_time) if prev_time else 0
```

### Visualization
Bounding boxes, labels, and FPS are drawn on each frame:
```python
cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
```

## Example Output

- Detected objects are shown with bounding boxes and unique track IDs.
- FPS is displayed on the top-left corner of the video frame.

## Notes

- Adjust the confidence threshold (`conf_threshold`) as needed:
  ```python
  conf_threshold = 0.5
  ```
- Modify `colors` for custom bounding box colors.

