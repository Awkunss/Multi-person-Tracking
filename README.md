
# YOLO Object Tracking with ByteTrack

This project implements object detection and tracking using YOLO and ByteTrack. The application supports video file input or live webcam feed.

---

## Project Structure

```
/tracking/
├── arguments.py          # Handles command-line arguments
├── main.py               # Entry point for the application
├── tracker.py            # Contains YOLOv9 tracker logic
├── video_processor.py    # Handles video input and processing
├── config.py             # Stores configuration constants
├── tracker_config.yaml   # Config tracker
```

---

## Features

- **Object Detection**: Uses YOLO for detecting objects in frames.
- **Object Tracking**: ByteTrack for tracking detected objects across frames.
- **Input Options**:
  - **Video File**: Process a pre-recorded video file.
  - **Webcam**: Stream and process live video feed.
- **Command-Line Arguments**: Easily specify input type and video path.

---

## Usage

Run the application using `main.py`. Specify the input type and video path (if applicable) via command-line arguments.

### Command-Line Arguments

| Argument      | Description                                    | Required |
|---------------|------------------------------------------------|----------|
|   `--input`   | Input Path to the video file or `0` for webcam | Yes      |

### Example Commands

#### Process a Video File
```bash
python main.py --input walking_2.mp4
```

#### Stream Live from Webcam
```bash
python main.py --input 0
```

---

## Configuration

You can modify constants like YOLO weights, tracker in `config.py`:

```python
# config.py
VERSION_MODEL = "yolo11m.pt"
TRACKER = "tracker_config.yaml" # Can be replaced with "bytetrack.yaml" or "botsort.yaml" built in ultralytics
```

---

## How It Works

1. **Detection**: YOLO identifies objects in each frame.
2. **Tracking**: ByteTrack assigns unique IDs to detected objects and tracks them across frames.
3. **Visualization**: Draws bounding boxes and displays the tracking IDs on the video.

---

## Output

- Displays real-time video feed with tracked objects.
- Press `q` to exit the application.

---

## Notes

- For the best performance, ensure CUDA is installed and available for PyTorch.


