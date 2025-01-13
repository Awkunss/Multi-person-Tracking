
# YOLOv9 Object Tracking with DeepSORT

This project implements object detection and tracking using YOLOv9 and DeepSORT. The application supports video file input or live webcam feed.

---

## Project Structure

```
yolov9/tracking/
├── arguments.py          # Handles command-line arguments
├── main.py               # Entry point for the application
├── tracker.py            # Contains YOLOv9 tracker logic
├── video_processor.py    # Handles video input and processing
├── config.py             # Stores configuration constants
```

---

## Features

- **Object Detection**: Uses YOLOv9 for detecting objects in frames.
- **Object Tracking**: Employs DeepSORT for tracking detected objects across frames.
- **Input Options**:
  - **Video File**: Process a pre-recorded video file.
  - **Webcam**: Stream and process live video feed.
- **Command-Line Arguments**: Easily specify input type and video path.

---

## Usage

Run the application using `main.py`. Specify the input type and video path (if applicable) via command-line arguments.

### Command-Line Arguments

| Argument      | Description                                   | Required |
|---------------|-----------------------------------------------|----------|
| `--input_type`| Input type: `1` for video file, `2` for webcam | Yes      |
| `--video_path`| Path to the video file (if `input_type=1`)    | No       |

### Example Commands

#### Process a Video File
```bash
python main.py --input_type 1 --video_path data_ext/walking_2.mp4
```

#### Stream Live from Webcam
```bash
python main.py --input_type 2
```

---

## Configuration

You can modify constants like YOLOv9 weights, class names path, and confidence threshold in `config.py`:

```python
# config.py
WEIGHTS_PATH = "weights/yolov9-s-converted.pt"
CLASSES_PATH = "data_ext/classes.names"
CONF_THRESHOLD = 0.5
```

---

## How It Works

1. **Detection**: YOLOv9 identifies objects in each frame.
2. **Tracking**: DeepSORT assigns unique IDs to detected objects and tracks them across frames.
3. **Visualization**: Draws bounding boxes and displays the tracking IDs on the video.

---

## Output

- Displays real-time video feed with tracked objects.
- Press `q` to exit the application.

---

## Folder Layout

Ensure the following folder layout before running the application:

```
yolov9/
├── weights/
│   └── yolov9-s-converted.pt    # YOLOv9 model weights
├── data_ext/
│   ├── classes.names            # Class names file
│   └── walking_2.mp4            # Example video file
```

---

## Notes

- For the best performance, ensure CUDA is installed and available for PyTorch.
- This project assumes a pre-trained YOLOv9 model in the `weights` folder.


