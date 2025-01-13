from arguments import parse_args
from tracker import YOLOv9Tracker
from video_processor import VideoProcessor
import config


def main():
    args = parse_args()

    if args.input_type == 1 and not args.video_path:
        raise ValueError("Error: --video_path is required when --input_type is 1.")

    # Initialize YOLOv9 Tracker
    yolo_tracker = YOLOv9Tracker(
        weights_path=config.WEIGHTS_PATH,
        classes_path=config.CLASSES_PATH,
        conf_threshold=config.CONF_THRESHOLD,
    )

    # Process Video
    video_processor = VideoProcessor(input_type=args.input_type, video_path=args.video_path)
    video_processor.process(yolo_tracker)


if __name__ == "__main__":
    main()
