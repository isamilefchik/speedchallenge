import cv2
import numpy as np

def parse_speeds():
    filepath = "./data/train.txt"
    with open(filepath) as file:
        raw = file.read()
    return raw.split("\n")

def get_next_frame(cap, prev_frame):
    _, cur = cap.read()
        
    hsv_flow = np.zeros_like(cur)
    hsv_flow[..., 1] = 255

    cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_frame, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv_flow[..., 0] = ang*180/np.pi/2
    hsv_flow[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb_flow = cv2.cvtColor(hsv_flow, cv2.COLOR_HSV2BGR)
    
    return rgb_flow, cur_gray