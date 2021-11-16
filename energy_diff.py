import cv2
import numpy as np
import argparse
SEAM_COLOR = np.array([255, 200, 200])


def rotate_image(image, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(image, k)


def visualize(im, boolmask=None, rotate=False):
    vis = im.astype(np.uint8)
    if boolmask is not None:
        vis[np.where(boolmask is False)] = SEAM_COLOR
    if rotate:
        vis = rotate_image(vis, False)
    # cv2.imshow("visualization", vis)
    # cv2.waitKey(1)
    return vis

def forward_energy(im):
    """
    Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
    by Rubinstein, Shamir, Avidan.
    Vectorized code adapted from
    https://github.com/axu2/improved-seam-carving.
    """
    h, w = im.shape[:2]
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))

    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)

    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

    for i in range(1, h):
        mU = m[i - 1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)

        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)

    # vis = visualize(energy)
    # cv2.imwrite("demo.jpg", vis)

    return energy


def main():
    args = parse.parse_args()
    video_path = args.video_path
    output_path = args.output_path

    cap = cv2.VideoCapture(video_path)
    ret, current_frame = cap.read()

    # Obtain frame size information using get() method
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width, frame_height)
    fps = cap.get(5)

    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, frame_size, False)

    while ret:
        energy = forward_energy(current_frame)
        output.write(energy.astype(np.uint8))

        ret, current_frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--video_path', type=str, default='you forgot to set video path')
    parse.add_argument('--output_path', type=str, default='you forgot to set output path')

    main()