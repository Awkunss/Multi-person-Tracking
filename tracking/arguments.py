import argparse


def parse_args():
    """
    Parse command-line arguments for the YOLOv9 tracking application.
    """
    parser = argparse.ArgumentParser(description="YOLOv9 Object Tracking")
    parser.add_argument(
        "--input_type", type=int, choices=[1, 2], required=True,
        help="Input type: 1 for video file, 2 for webcam."
    )
    parser.add_argument(
        "--video_path", type=str, default=None,
        help="Path to the video file (required if input_type is 1)."
    )
    return parser.parse_args()
