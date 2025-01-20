import cv2
import time

import torch
from ultralytics import YOLO


class YOLOVideoTracker:
    def __init__(self, model_version, video_path, tracker_config, target_classes=[0]):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load the YOLO model
        self.model = YOLO(model_version).to(self.device)
        self.video_path = video_path
        self.tracker_config = tracker_config
        self.target_classes = target_classes
        self.cap = self.initialize_video_capture(video_path)

    def initialize_video_capture(self, video_path):
        if video_path is not None:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(0)  # 0 for webcam
        if not cap.isOpened():
            raise ValueError("Error: Unable to open video source.")
        return cap

    def process_frame(self, frame, prev_time):
        """Process a single frame, run YOLO tracking, and annotate it."""
        # Run YOLO tracking on the frame
        results = self.model.track(frame, persist=True, tracker=self.tracker_config, classes=self.target_classes)
        annotated_frame = results[0].plot()

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0

        # Draw FPS on the frame
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return annotated_frame, curr_time

    def track_video(self):
        """Process the video frame by frame."""
        prev_time = time.time()

        while self.cap.isOpened():
            # Read a frame from the video
            success, frame = self.cap.read()
            
            if success:
                # Process the frame and get annotated results
                annotated_frame, prev_time = self.process_frame(frame, prev_time)

                # Display the annotated frame
                height, width = annotated_frame.shape[:2]
                annotated_frame = cv2.resize(annotated_frame, (int(width/2), int(height/2)))
                cv2.imshow("YOLO Tracking", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                print("End of video reached or unable to read frame.")
                break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()