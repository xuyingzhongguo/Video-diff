import cv2
import argparse
import numpy as np
from utils import *


def main():
    args = parse.parse_args()
    video_path = args.video_path
    output_path = args.output_path
    channel_num = args.channel_num
    method = args.method
    thresh = args.threshold
    thresh_val = args.threshold_val

    cap = cv2.VideoCapture(video_path)
    ret, current_frame = cap.read()
    previous_frame = current_frame

    # Obtain frame size information using get() method
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width, frame_height)
    fps = cap.get(5)

    # Initialize video writer object
    if channel_num == 3:
        output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, frame_size)
    elif channel_num == 1:
        output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, frame_size, False)
    else:
        sys.exit('Wrong channel number')

    while ret:
        frame_diff = cal_diff(method, current_frame, previous_frame, thresh, thresh_val)
        output.write(frame_diff)

        previous_frame = current_frame.copy()
        ret, current_frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--video_path', type=str, default='you forgot to set video path')
    parse.add_argument('--output_path', type=str, default='you forgot to set output path')
    parse.add_argument('--method', type=str, default='abs')
    parse.add_argument('--channel_num', type=int, default=3)
    parse.add_argument('--threshold', action='store_true')
    parse.add_argument('--threshold_val', type=int, default=127)

    main()

