import cv2


class VideoProcessor:
    def __init__(self, video_path=None):
        self.cap = self._initialize_capture(video_path)

    def _initialize_capture(self, video_path):
        if video_path is not None:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(0)  # 0 for webcam
        if not cap.isOpened():
            raise ValueError("Error: Unable to open video source.")
        return cap

    def process(self, yolo_tracker):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            processed_frame = yolo_tracker.process_frame(frame)

            height, width = processed_frame.shape[:2]
            display_frame = cv2.resize(processed_frame, (int(width/2), int(height/2)))
            cv2.imshow('YOLOv9 Tracker', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
