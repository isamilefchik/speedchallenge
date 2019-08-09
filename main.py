#!/usr/local/bin/python3

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
from model import Speed_Classify_Model, train_model
from load_data import parse_speeds, get_next_frame

def main():
    parser = argparse.ArgumentParser(description="Speedchallenge solver")
    parser.add_argument("-v", "--visualize", action="store_true", default=False, help="Do live visualization while training.")
    parser.add_argument("-l", "--load", type=str, default="", help="Path to saved model, if model should be loaded.")
    parser.add_argument("-s", "--save", type=str, default="", help="Name of model if model should be saved at each epoch.")

    args = parser.parse_args()
    live_viz, load_path, model_name = args.visualize, args.load, args.save

    if live_viz:
        plt.ion()
        fig = plt.figure()

    ground_truth = parse_speeds()
    model = Speed_Classify_Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 2

    if load_path != "":
        model.load_state_dict(torch.load(load_path, map_location='cpu'))

    for epoch in range(num_epochs):
        rand_indices = np.random.permutation(np.arange(1,20399))
        results = []
        running_loss = 0

        for count, i in enumerate(rand_indices):
            img = cv2.imread("./data/train_frames/" + str(i) + ".jpg")
            speed = ground_truth[i]


            if live_viz:
                cv2.imshow('Current Frame', img)
                k = cv2.waitKey(5)
                yes = False
                if k == ord('p'):
                    yes = True
                if k == 27: # ESC key
                    break


            train_loss, prediction_array = train_model(model, optimizer, img, speed)
            running_loss += train_loss

            possible_speeds = np.arange(90)
            prediction = np.average(possible_speeds, weights=prediction_array)
            
            if live_viz:
                plt.cla()
                plt.bar(np.arange(0,90), prediction_array)
                fig.canvas.start_event_loop(0.0001)

            # First frame's prediction is equal to second frame's
            if i == 0:
                results.append(prediction)
            results.append(prediction)
            
            if count+1 != 20398:
                print("e{0}:{1}/20398 True Speed: {2}    Predicted Speed: {3:.6f}    Loss: {4:.6f}    ".format(epoch, count+1, speed, prediction, train_loss), end='\r')
            else:
                print("e{0}:{1}/20398 True Speed: {2}    Predicted Speed: {3:.6f}    Loss: {4:.6f}    ".format(epoch, count+1, speed, prediction, train_loss))
        
        print("===== Epoch {0}\tTotal Loss: {1:.6f} =====".format(epoch, running_loss))

        if model_name != "":
            save_path = "./model_saves/" + model_name + "_e" + str(epoch) + ".pth"
            torch.save(model.state_dict(), save_path)
    
    if live_viz:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
