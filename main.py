
from arguments import parse_args
from video_processing import VideoProcessor
from config import VERSION_MODEL, TRACKER

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Process Video
    if args.input == "0":
        video_processor = VideoProcessor(model_version=VERSION_MODEL,tracker_config=TRACKER, video_path=None)
    else:
        video_processor = VideoProcessor(model_version=VERSION_MODEL,tracker_config=TRACKER, video_path=args.input)

    video_processor.process_video()
