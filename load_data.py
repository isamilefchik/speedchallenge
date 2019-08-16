import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np

def parse_speeds():
    """ Reads ground truth speeds as float values. """
    filepath = "./data/train.txt"
    with open(filepath) as file:
        raw = file.read()
    result = list(map(float, raw.split("\n")))
    return result

######################################################
#
# Unneeded at current moment:
# ---------------------------
#
# def get_next_frame(cap, prev_frame):
    # success, cur = cap.read()
    # if not success:
        # sys.exit()
        
    # hsv_flow = np.zeros_like(cur)
    # hsv_flow[..., 1] = 255

    # cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

    # flow = cv2.calcOpticalFlowFarneback(prev_frame, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # hsv_flow[..., 0] = ang*180/np.pi/2
    # scale_cap = 80.
    # mag_clips = mag > scale_cap
    # mag[mag_clips] = scale_cap
    # mag = mag * (255. / scale_cap)
    # hsv_flow[..., 2] = mag

    # rgb_flow = cv2.cvtColor(hsv_flow, cv2.COLOR_HSV2BGR)
    
    # return rgb_flow, cur_gray


def plot_mag_distribution():
    """ This function helped me develop a sense of the distribution of
    magnitudes of the optical flow calculations. I used this information
    to normalize across the entire training dataset.
    """

    cap = cv2.VideoCapture("./data/train.mp4")

    success, prev_frame = cap.read()
    if not success:
        sys.exit()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    success, cur_frame = cap.read()

    count = 1
    mags = []
    while(success):
        hsv_flow = np.zeros_like(cur_frame)
        hsv_flow[..., 1] = 255

        cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_frame, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        mags.append(int(np.max(mag))) 

        print("{0} mags read        ".format(count), end="\r")
        prev_frame = cur_gray
        success, cur_frame = cap.read()
        count += 1

    print("Max max-value: {}     Min max-value: {}".format(np.max(mags), np.min(mags)))
    plt.style.use("ggplot")
    plt.hist(mags, bins=np.arange(600, step=5))
    plt.yscale("log")
    plt.show()

# plot_mag_distribution()



