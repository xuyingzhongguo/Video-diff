import cv2
import numpy as np
import sys
SEAM_COLOR = np.array([255, 200, 200])


def cal_diff(method, current_frame, previous_frame, thresh, thresh_val):
    if method == 'abs':
        frame_diff = cv2.absdiff(current_frame, previous_frame)
    elif method == 'sub':
        frame_diff = current_frame - previous_frame
    elif method == 'inverse_add':
        previous_frame_new = 255 - previous_frame
        frame_diff = current_frame / 2 + previous_frame_new / 2
        frame_diff = frame_diff.astype(np.uint8)
        frame_diff = np.minimum(255, frame_diff*10)
    elif method == 'amplify':
        frame_diff = np.min(255, cv2.absdiff(current_frame, previous_frame))

    # single channel
    elif method == 'rgb_g':
        frame_diff = cv2.absdiff(current_frame[:, :, 1], previous_frame[:, :, 1])
    elif method == 'rgb_g_thresh':
        frame_diff = cv2.absdiff(current_frame[:, :, 1], previous_frame[:, :, 1])
    elif method == 'ycrcb_y':
        ycbcr_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2YCrCb)
        ycbcr_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2YCrCb)
        frame_diff = cv2.absdiff(ycbcr_current[:, :, 0], ycbcr_previous[:, :, 0])
    elif method == 'hsv_v':
        ycbcr_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        ycbcr_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2HSV)
        frame_diff = cv2.absdiff(ycbcr_current[:, :, 2], ycbcr_previous[:, :, 2])

    else:
        sys.exit("Your diff method is not found")

    print(thresh)
    if thresh:
        ret, frame_diff_final = cv2.threshold(frame_diff, thresh_val, 255, cv2.THRESH_BINARY)

    return frame_diff_final


# Energy
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
