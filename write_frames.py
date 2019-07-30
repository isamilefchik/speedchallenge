import cv2
from load_data import get_next_frame
vidcap = cv2.VideoCapture('./data/train.mp4')
success, frame1 = vidcap.read()
count = 0
prev_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
while success:
    cur_flow, cur_frame = get_next_frame(vidcap, prev_frame)
    cv2.imwrite("./data/frames/%d.jpg" % count, cur_flow)
    prev_frame = cur_frame
    print('Read a new frame: {} {}         '.format(success, count), end='\r')
    count += 1