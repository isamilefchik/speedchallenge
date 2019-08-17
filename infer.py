#!/usr/local/bin/python3
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import argparse
from os import path
from os import listdir 
from model import Speed_Classify_Model, train_model
from load_data import parse_speeds 

def main():
    """ Main routine. """

    parser = argparse.ArgumentParser(description="Speedchallenge Inference")
    parser.add_argument("-i", "--input", type=str, default="train",
                        help="Which dataset (\"train\" or \"test\").")
    parser.add_argument("-m", "--mask", action="store_true", default=False,
                        help="Use mask.")
    parser.add_argument("-v", "--visualize", action="store_true",
                        default=False, help="Do live visualization.")
    parser.add_argument("-l", "--load", type=str, default="", help="Path to model.")
    parser.add_argument("-o", "--output", type=str, default="model_output.txt",
                        help="Path to text file to output results.")
    parser.add_argument("-s", "--simple", action="store_true", default=False,
                        help="Do a simple print (pure m/s measurements).")

    args = parser.parse_args()
    input_set, live_viz, load_path = args.input, args.visualize, args.load
    save_path, simple, use_mask = args.output, args.simple, args.mask

    assert input_set == "train" or input_set == "test", "Invalid input argument."
    assert path.exists(load_path), "Model does not exists."

    if live_viz:
        plt.ion()
        fig = plt.figure()
        plt.style.use("ggplot")

    model = Speed_Classify_Model()
    model.load_state_dict(torch.load(load_path, map_location='cpu'))

    if use_mask:
        mask = cv2.imread("./data/mask.png", 0)

    model.eval()

    results = []

    if live_viz:
        cap = cv2.VideoCapture("./data/" + input_set + ".mp4")
        cap.read()

    file_nums = []
    for cur_file in listdir("./data/better_" + input_set + "_frames/"):
        if cur_file[0] != "t":
            if len(cur_file.split(".")) == 2 and cur_file.split(".")[1] == "jpg":
                file_nums.append(int(cur_file.split(".")[0]))
    num_frames = max(file_nums)
        
    for i in range(1, num_frames+1):
        if live_viz:
            _, cur_frame = cap.read()
            cv2.imshow('Current Frame', cur_frame)
            k = cv2.waitKey(5)
            if k == 27: # ESC key
                break

        flow_img = cv2.imread("./data/better_" + input_set + "_frames/" + str(i) + ".jpg")

        if use_mask:
            flow_img = cv2.bitwise_and(flow_img, flow_img, mask=mask) 

        transform = torchvision.transforms.ToTensor()
        flow_tensor = transform(flow_img)
        flow_tensor = torch.unsqueeze(flow_tensor, 0)

        result = model(flow_tensor).detach().numpy()
        possible_speeds = np.arange(90)
        prediction = np.average(possible_speeds, weights=result)

        if live_viz:
            plt.cla()
            plt.bar(np.arange(0,90), result)
            fig.canvas.start_event_loop(0.0001)

        # First frame's prediction is equal to second frame's
        if i == 1:
            results.append(prediction)
            results.append(prediction)
        elif i == 2:
            results.append((results[-1] + prediction) / 2.)
        elif i == 3:
            results.append((results[-2] + results[-1] + prediction) / 3.)
        else:
            results.append((results[-3] + results[-2] + results[-1] + prediction) / 4.)
        
        if i != num_frames and i % 20 == 0:
            print("{0}/{1} Predicted Speed: {2:.4f}    ".format(i, num_frames, prediction), end='\r')
        elif i == num_frames:
            print("{0}/{1} Predicted Speed: {2:.4f}    ".format(i, num_frames, prediction))

    ground_truth = parse_speeds()
    with open(save_path, 'w') as f:
        if simple:
            for i, item in enumerate(results):
                if i != len(results) - 1:
                    f.write(str(item) + "\n")
                else:
                    f.write(str(item))
        else:
            if input_set == "train": 
                for i, item in enumerate(results):
                    true_speed = ground_truth[i]
                    print_str = "True: {0:.2f} m/s | {1:.2f} mph ".format(true_speed, true_speed * 2.23694) \
                        + "--- Model: {2:.2f} m/s | {3:.2f} mph".format(item, item * 2.23694)
                    if i != len(results) - 1:
                        f.write(print_str + "\n")
                    else:
                        f.write(print_str)
            else:
                for i, item in enumerate(results):
                    print_str = "Model: {0:.4f} m/s | {1:.2f} mph".format(item, item * 2.23694)
                    if i != len(results) - 1:
                        f.write(print_str + "\n")
                    else:
                        f.write(print_str)

if __name__ == "__main__":
    main()
