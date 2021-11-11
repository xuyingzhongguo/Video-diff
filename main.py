import cv2
import argparse


def main():
    args = parse.parse_args()
    video_path = args.video_path
    output_path = args.output_path

    cap = cv2.VideoCapture(video_path)
    ret, current_frame = cap.read()
    previous_frame = current_frame

    # Obtain frame size information using get() method
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width, frame_height)
    fps = cap.get(5)

    # Initialize video writer object
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, frame_size)

    while ret:
        # current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        # previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        #
        # frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)

        frame_diff = cv2.absdiff(current_frame, previous_frame)

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

    main()

