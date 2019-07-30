import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import parse_speeds, get_next_frame


def main():
    ground_truth = parse_speeds()

    cap = cv2.VideoCapture("./data/train.mp4")

    _, frame1 = cap.read()
    prev_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    while(1):
        cur_flow, cur_frame = get_next_frame(cap, prev_frame)
        # cv2.imshow('frame2', cur_flow)

        # k = cv2.waitKey(5)
        # if k == 27: # ESC key
        #     break

        
        
        prev_frame = cur_frame

    cap.release()
    # cv2.destroyAllWindows()



if __name__ == "__main__":
    main()