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
from load_data import parse_speeds, get_next_frame

def main():
    parser = argparse.ArgumentParser(description="Speedchallenge tester")
    parser.add_argument("-i", "--input", type=str, default="train", help="Which dataset (\"train\" or \"test\").")
    parser.add_argument("-v", "--visualize", action="store_true", default=False, help="Do live visualization.")
    parser.add_argument("-l", "--load", type=str, default="", help="Path to model.")
    parser.add_argument("-o", "--output", type=str, default="model_output.txt", help="Path to text file to output results.")

    args = parser.parse_args()
    input_set, live_viz, load_path, save_path = args.input, args.visualize, args.load, args.output

    assert input_set == "train" or input_set == "test", "Invalid input argument."
    assert path.exists(load_path), "Model does not exists."

    if live_viz:
        plt.ion()
        fig = plt.figure()
        plt.style.use("ggplot")

    model = Speed_Classify_Model()

    model.load_state_dict(torch.load(load_path, map_location='cpu'))

    model.eval()
    # loss = nn.BCELoss()

    results = []

    if live_viz:
        cap = cv2.VideoCapture("./data/" + input_set + ".mp4")
        cap.read()

    file_nums = []
    for cur_file in listdir("./data/" + input_set + "_frames/"):
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

        flow_img = cv2.imread("./data/" + input_set + "_frames/" + str(i) + ".jpg")
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
        if i == 0:
            results.append(prediction)
        results.append(prediction)
        
        if i != num_frames and i % 20 == 0:
            print("{0}/{1} Predicted Speed: {2:.4f}    ".format(i, num_frames, prediction), end='\r')
        elif i == num_frames:
            print("{0}/{1} Predicted Speed: {2:.4f}    ".format(i, num_frames, prediction))

    ground_truth = parse_speeds()
    with open(save_path, 'w') as f:
        if input_set == "train": 
            for i, item in enumerate(results):
                if i != len(results) - 1:
                    f.write("True: {0:.4f} --- Model: {1:.4f}\n".format(ground_truth[i], item))
                else:
                    f.write("True: {0:.4f} --- Model: {1:.4f}".format(ground_truth[i], item))
        else:
            for i, item in enumerate(results):
                if i != len(results) - 1:
                    f.write("Model: {0:.4f}\n".format(item))
                else:
                    f.write("Model: {0:.4f}".format(item))

if __name__ == "__main__":
    main()
