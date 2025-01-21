import time

import cv2
import torch
from ultralytics import YOLO


class YOLOVideoTracker:
    def __init__(self, model_version,  tracker_config, target_classes=[0]):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load the YOLO model
        self.model = YOLO(model_version).to(self.device)
        self.tracker_config = tracker_config
        self.target_classes = target_classes
        

    def __initialize_video_capture(self, video_path):
        if video_path is not None:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(0)  # 0 for webcam
        if not cap.isOpened():
            raise ValueError("Error: Unable to open video source.")
        return cap

    def __process_frame(self, __process_frame, __prev_time):
        """Process a single frame, run YOLO tracking, and annotate it."""
        # Run YOLO tracking on the frame
        results = self.model.track(
            __process_frame, 
            persist=True, 
            tracker=self.tracker_config, 
            classes=self.target_classes,
        )   
        __annotated_frame = results[0].plot()

        # Calculate FPS
        __curr_time = time.time()
        fps = 1 / (__curr_time - __prev_time) if __curr_time != __prev_time else 0

        # Draw FPS on the frame
        cv2.putText(__annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return __annotated_frame, __curr_time

    def track_video(self, video_path):
        """Process the video frame by frame."""
        __prev_time = time.time()

        cap = self.__initialize_video_capture(video_path)
        while cap.isOpened():
            # Read a frame from the video
            success, __process_frame = cap.read()
            
            if success:
                # Process the frame and get annotated results
                __annotated_frame, __prev_time = self.__process_frame(__process_frame, __prev_time)

                # Display the annotated frame
                __height, __width = __annotated_frame.shape[:2]
                __annotated_frame = cv2.resize(__annotated_frame, (int(__width/2), int(__height/2)))
                cv2.imshow("YOLO Tracking", __annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                print("End of video reached or unable to read frame.")
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()