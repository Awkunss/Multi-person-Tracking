import argparse


def parse_args():
    """
    Parse command-line arguments for the YOLOv9 tracking application.
    """
    parser = argparse.ArgumentParser(description="YOLOv9 Object Tracking")

    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to the video file or '0' for webcam"
    )
    return parser.parse_args()
