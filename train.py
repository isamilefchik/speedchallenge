#!/usr/local/bin/python3

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import time
import datetime
from model import Speed_Classify_Model, train_model
from load_data import parse_speeds, get_next_frame

def main():
    parser = argparse.ArgumentParser(description="Speedchallenge train")
    parser.add_argument("-v", "--visualize", action="store_true", default=False, help="Do live visualization while training.")
    parser.add_argument("-l", "--load", type=str, default="", help="Path to saved model, if model should be loaded.")
    parser.add_argument("-s", "--save", type=str, default="", help="Name of model if model should be saved at each epoch.")
    parser.add_argument("-e", "--epochs", type=int, default=2, help="Number of epochs to train.")

    args = parser.parse_args()
    live_viz, load_path, model_name, num_epochs = args.visualize, args.load, args.save, args.epochs

    if live_viz:
        plt.ion()
        fig = plt.figure()

    ground_truth = parse_speeds()
    model = Speed_Classify_Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    mask = cv2.imread("./data/mask.png", 0)

    if load_path != "":
        model.load_state_dict(torch.load(load_path, map_location='cpu'))
        prev_epoch = int(load_path[-5])
        epoch_list = range(prev_epoch + 1, num_epochs + prev_epoch + 1)
    else:
        epoch_list = range(num_epochs)

    for epoch in epoch_list:
        rand_indices = np.random.permutation(np.arange(1,20400))
        # results = []
        running_loss = 0
        epoch_size = 8000
        prev_remain_seconds = 1e10 
        start_time = time.time()
        refresh_time = time.time()
        dt_indices = 0

        for count, i in enumerate(rand_indices[0:epoch_size]):
            img = cv2.imread("./data/better_train_frames/" + str(i) + ".jpg")
            img = cv2.bitwise_and(img, img, mask=mask) 

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
            # if i == 0:
                # results.append(prediction)
            # results.append(prediction)
            
            dt_indices += 1
            check_time = time.time()

            if (round(check_time) - round(refresh_time) >= 1) or count+1 == epoch_size:
                dt = check_time - start_time
                remain_dt_indices = (float(epoch_size - (count+1)) / dt_indices) + ((epoch_list[-1] - epoch) * (epoch_size / dt_indices))
                remain_seconds = round(remain_dt_indices * dt)
                est = str(datetime.timedelta(seconds=remain_seconds))
                
                if count+1 != epoch_size:
                    print_str_1 = "e{0}:{1}/{2} ".format(epoch, count+1, epoch_size)
                    print_str_2 = "True Speed: {0:.2f}   ".format(speed)
                    print_str_3 = "Predicted Speed: {0:.2f}   ".format(prediction)
                    print_str_4 = "Loss: {0:.2f}   ".format(train_loss)
                    print_str_5 = "est: {0}    ".format(est)
                    print(print_str_1 + print_str_2 + print_str_3 + print_str_4 + print_str_5, end='\r')
                    prev_remain_seconds = remain_seconds
                else:
                    print_str_1 = "e{0}:{1}/{2} ".format(epoch, count+1, epoch_size)
                    print_str_2 = "True Speed: {0:.2f}   ".format(speed)
                    print_str_3 = "Predicted Speed: {0:.2f}   ".format(prediction)
                    print_str_4 = "Loss: {0:.2f}   ".format(train_loss)
                    print_str_5 = "est: {0}    ".format(est)
                    print(print_str_1 + print_str_2 + print_str_3 + print_str_4 + print_str_5)

                refresh_time = time.time()

        print("===== Epoch {0}\tTotal Loss: {1:.6f} =====".format(epoch, running_loss))

        if model_name != "":
            save_path = "./model_saves/" + model_name + "_e" + str(epoch) + ".pth"
            torch.save(model.state_dict(), save_path)
    
    if live_viz:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
