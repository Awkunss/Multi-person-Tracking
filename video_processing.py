from tracker import YOLOVideoTracker

class VideoProcessor:
    def __init__(self, model_version, video_path, tracker_config, target_classes=[0]):
        self.tracker = YOLOVideoTracker(model_version, video_path, tracker_config, target_classes)

    def process_video(self):
        self.tracker.track_video()