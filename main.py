import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import Speed_Classify_Model, train_model
from load_data import parse_speeds, get_next_frame

def main():
    # plt.ion()
    # fig = plt.figure()

    ground_truth = parse_speeds()
    model = Speed_Classify_Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 10

    LOAD_MODEL_PATH = "./model_saves/model_e1.pth"
    model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location='cpu'))

    for epoch in range(num_epochs):
        rand_indices = np.random.permutation(np.arange(1,20399))
        results = []
        running_loss = 0

        for count, i in enumerate(rand_indices):
            # cv2.imshow('Current Frame', cur_flow)
            # k = cv2.waitKey(5)
            # yes = False
            # if k == ord('p'):
            #     yes = True
            # if k == 27: # ESC key
            #     break

            img = cv2.imread("./data/frames/" + str(i) + ".jpg")
            speed = ground_truth[i]

            train_loss, prediction_array = train_model(model, optimizer, img, speed)
            running_loss += train_loss

            prediction = np.argmax(prediction_array)

            # plt.cla()
            # plt.bar(np.arange(0,90), prediction_array)
            # fig.canvas.start_event_loop(0.0001)

            # First frame's prediction is equal to second frame's
            if i == 0:
                results.append(prediction)
            results.append(prediction)
            
            if count+1 != 20398:
                print("e{0}:{1}/20398 True Speed: {2}\tPredicted Speed: {3}\tLoss: {4:.6f}    ".format(epoch, count+1, speed, prediction, train_loss), end='\r')
            else:
                print("e{0}:{1}/20398 True Speed: {2}\tPredicted Speed: {3}\tLoss: {4:.6f}    ".format(epoch, count+1, speed, prediction, train_loss))
        
        print("===== Epoch {0}\tTotal Loss: {1:.6f} =====".format(epoch, running_loss))
        save_path = "./model_saves/model2" + "_e" + str(epoch) + ".pth"
        torch.save(model.state_dict(), save_path)
        
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()