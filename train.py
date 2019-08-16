#!/usr/local/bin/python3

import argparse
import time
import datetime
from os import listdir
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import Speed_Classify_Model, train_model
from load_data import parse_speeds

def main():
    """ Main routine. """

    parser = argparse.ArgumentParser(description="Speedchallenge Trainer")
    parser.add_argument("-v", "--visualize", action="store_true",
                        default=False, help="Do live visualization while training.")
    parser.add_argument("-l", "--load", type=str, default="",
                        help="Path to saved model, if model should be loaded.")
    parser.add_argument("-s", "--save", type=str, default="",
                        help="Name of model if model should be saved at each epoch.")
    parser.add_argument("-e", "--epochs", type=int, default=2,
                        help="Number of epochs to train.")

    args = parser.parse_args()
    live_viz, load_path, model_name, num_epochs = args.visualize, args.load, args.save, args.epochs

    if live_viz:
        plt.ion()
        fig = plt.figure()

    ground_truth = parse_speeds()
    model = Speed_Classify_Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # mask = cv2.imread("./data/mask.png", 0)

    if load_path != "":
        model.load_state_dict(torch.load(load_path, map_location='cpu'))
        prev_epoch = int(load_path[-5])
        epoch_list = range(prev_epoch + 1, num_epochs + prev_epoch + 1)
    else:
        epoch_list = range(num_epochs)

    file_nums = []
    for cur_file in listdir("./data/better_train_frames/"):
        if len(cur_file.split(".")) == 2 and cur_file.split(".")[1] == "jpg":
            file_nums.append(int(cur_file.split(".")[0]))
                
    max_frame_num = max(file_nums)

    for epoch in epoch_list:
        rand_indices = np.random.permutation(np.arange(1, max_frame_num+1))
        running_loss = 0
        epoch_size = 8000
        start_time = time.time()
        refresh_time = time.time()
        dt_indices = 0

        for count, i in enumerate(rand_indices[0:epoch_size]):
            img = cv2.imread("./data/better_train_frames/" + str(i) + ".jpg")
            # img = cv2.bitwise_and(img, img, mask=mask)

            if i < 20400:
                speed = ground_truth[i]
            else:
                speed = 0.0

            if live_viz:
                cv2.imshow('Current Frame', img)
                k = cv2.waitKey(5)
                if k == 27:
                    break

            train_loss, prediction_array = train_model(model, optimizer, img, speed)
            running_loss += train_loss

            possible_speeds = np.arange(90)
            prediction = np.average(possible_speeds, weights=prediction_array)

            if live_viz:
                plt.cla()
                plt.bar(np.arange(0, 90), prediction_array)
                fig.canvas.start_event_loop(0.0001)

            dt_indices += 1
            check_time = time.time()

            # Status printing
            if (check_time - refresh_time >= 1) or count+1 == epoch_size:
                delta_t = check_time - start_time
                remain_dt_indices = (float(epoch_size - (count+1)) / dt_indices) + \
                                    ((epoch_list[-1] - epoch) * (epoch_size / dt_indices))
                remain_seconds = round(remain_dt_indices * delta_t)
                est = str(datetime.timedelta(seconds=remain_seconds))

                print_str_1 = "e{0}:{1}/{2} ".format(epoch, count+1, epoch_size)
                print_str_2 = "True Speed: {0:.2f}   ".format(speed)
                print_str_3 = "Predicted Speed: {0:.2f}   ".format(prediction)
                print_str_4 = "Loss: {0:.2f}   ".format(train_loss)
                print_str_5 = "est: {0}    ".format(est)

                if count+1 != epoch_size:
                    print(print_str_1 + print_str_2 + print_str_3 + print_str_4 + print_str_5, end='\r')
                else:
                    print(print_str_1 + print_str_2 + print_str_3 + print_str_4 + print_str_5)

                refresh_time = time.time()

        # for i in range(1120, 1601):
            # img = cv2.imread("./data/better_train_frames/t" + str(i) + ".jpg")

            # speed = 0.0
            # train_loss, prediction_array = train_model(model, optimizer, img, speed)
            # running_loss += train_loss

            # possible_speeds = np.arange(90)
            # prediction = np.average(possible_speeds, weights=prediction_array)

            # if i != 1600:
                # print("Special case prediction: " + str(prediction), end='\r')
            # else:
                # print("Special case prediction: " + str(prediction))
        

        print("===== Epoch {0}\tTotal Loss: {1:.6f} =====".format(epoch, running_loss))

        if model_name != "":
            save_path = "./model_saves/" + model_name + "_e" + str(epoch) + ".pth"
            torch.save(model.state_dict(), save_path)

    if live_viz:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
