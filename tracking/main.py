import sys

sys.path.insert(0, ".")

from arguments import parse_args

import config
from tracker import YOLOv9Tracker
from video_processor import VideoProcessor


def main():
    args = parse_args()

    # Initialize YOLOv9 Tracker
    yolo_tracker = YOLOv9Tracker(
        weights_path=config.WEIGHTS_PATH,
        classes_path=config.CLASSES_PATH,
        conf_threshold=config.CONF_THRESHOLD,
    )

    # Process Video
    if args.input == "0":
        video_processor = VideoProcessor()
    else:
        video_processor = VideoProcessor(video_path=args.input)
    video_processor.process(yolo_tracker)


if __name__ == "__main__":
    main()
