from arguments import parse_args

from config import  TRACKER, VERSION_MODEL
from tracker import YOLOVideoTracker


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Process Video
    video_path = None if args.input == "0" else args.input

    video_processor = YOLOVideoTracker(model_version=VERSION_MODEL,tracker_config=TRACKER) 
    video_processor.track_video(video_path)
