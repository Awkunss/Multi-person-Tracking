from tracker import YOLOVideoTracker
from config import VERSION_MODEL, TRACKER

class VideoProcessor:
    def __init__(self, video_path):
        self.tracker = YOLOVideoTracker(
            model_version=VERSION_MODEL,
            tracker_config=TRACKER,
            video_path = video_path
        )

    def process_video(self):
        self.tracker.track_video()