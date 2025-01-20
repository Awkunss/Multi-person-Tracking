
from arguments import parse_args
from video_processing import VideoProcessor

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Process Video
    video_path = None if args.input == "0" else args.input

    video_processor = VideoProcessor(
        video_path=video_path
    )    

    video_processor.process_video()
