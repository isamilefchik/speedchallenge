import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import Speed_Classify_Model, train_model
from load_data import parse_speeds, get_next_frame


def main():
    ground_truth = parse_speeds()
    model = Speed_Classify_Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    cap = cv2.VideoCapture("./data/train.mp4")

    _, frame1 = cap.read()
    prev_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    results = []

    for speed in ground_truth[1:]:
        cur_flow, cur_frame = get_next_frame(cap, prev_frame)
        cv2.imshow('frame2', cur_flow)

        k = cv2.waitKey(5)
        yes = False
        if k == ord('p'):
            yes = True
        if k == 27: # ESC key
            break

        train_loss, prediction_array = train_model(model, optimizer, cur_flow, speed, yes)
        prediction = np.argmax(prediction_array)
        results.append(prediction)
        print("True Speed: {}".format(speed))
        print("Predicted Speed: {}".format(prediction))
        print("Loss: {}".format(train_loss))
        
        prev_frame = cur_frame

    cap.release()
    # cv2.destroyAllWindows()



if __name__ == "__main__":
    main()