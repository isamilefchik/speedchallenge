#!/usr/local/bin/python3

import cv2

vidcap = cv2.VideoCapture('./data/train.mp4')
count = 1
success, frame = vidcap.read()
while success:
    cv2.imwrite("./data/pure_train_frames/%d.jpg" % count, frame)
    count += 1 
    success, frame = vidcap.read()
