import cv2

def main():
    vidcap = cv2.VideoCapture('./data/test.mp4')
    success, frame1 = vidcap.read()
    count = 1
    prev_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    while success:
        cur_flow, cur_frame = get_next_frame(vidcap, prev_frame, 70)
        cv2.imwrite("./data/better_test_frames/%d.jpg" % count, cur_flow)
        prev_frame = cur_frame
        if count % 10 == 0:
            print('Read a new frame: {} {}         '.format(success, count), end='\r')
        count += 1

def get_next_frame(cap, prev_frame, clip):
    success, cur = cap.read()
    if not success:
        sys.exit()
        
    hsv_flow = np.zeros_like(cur)
    hsv_flow[..., 1] = 255

    cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_frame, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv_flow[..., 0] = ang*180/np.pi/2
    # hsv_flow[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag_clips = mag > clip
    mag[mag_clips] = clip
    mag = mag * (255. / clip)
    hsv_flow[..., 2] = mag

    rgb_flow = cv2.cvtColor(hsv_flow, cv2.COLOR_HSV2BGR)
    
    return rgb_flow, cur_gray

