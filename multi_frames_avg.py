import cv2
import argparse
import numpy as np


def main():
    args = parse.parse_args()
    video_path = args.video_path
    output_path = args.output_path
    avg_num = args.avg_num

    cap = cv2.VideoCapture(video_path)
    ret, current_frame = cap.read()
    previous_frame = current_frame

    # Obtain frame size information using get() method
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # # Initialize video writer object
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, frame_size)

    for i in range(avg_num, frame_count-1):
        frame_sum = np.zeros((frame_height, frame_width, 3))
        for j in range(avg_num):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i - j)
            ret, frame = cap.read()
            frame_sum = frame_sum + frame

        frame_avg = (frame_sum/avg_num).astype(np.uint8)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i+1)
        ret, frame_now = cap.read()
        frame_output = cv2.absdiff(frame_now, frame_avg)
        output.write(frame_output)


    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--video_path', type=str, default='you forgot to set video path')
    parse.add_argument('--output_path', type=str, default='you forgot to set output path')
    parse.add_argument('--avg_num', type=int, default=5)

    main()

