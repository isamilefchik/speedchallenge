import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from model import Speed_Classify_Model, train_model
from load_data import parse_speeds, get_next_frame

# plt.ion()
# fig = plt.figure()
# plt.style.use("ggplot")

# ground_truth = parse_speeds()
model = Speed_Classify_Model()

LOAD_MODEL_PATH = "./model_saves/model_e1.pth"
model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location='cpu'))

model.eval()
# loss = nn.BCELoss()

results = []

# cap = cv2.VideoCapture("./data/test.mp4")
# cap.read()

for i in range(1, 10797):
    # _, cur_frame = cap.read()
    # cv2.imshow('Current Frame', cur_frame)
    # k = cv2.waitKey(5)
    # if k == 27: # ESC key
    #     break

    flow_img = cv2.imread("./data/test_frames/" + str(i) + ".jpg")
    transform = torchvision.transforms.ToTensor()
    flow_tensor = transform(flow_img)
    flow_tensor = torch.unsqueeze(flow_tensor, 0)

    # speed = ground_truth[i]
    # target_tensor = np.zeros((90), dtype=np.float32)
    # target_tensor[speed] = 1.0
    # target_tensor = torch.tensor(target_tensor)

    result = model(flow_tensor)
    # test_loss = loss(result, target_tensor)
    prediction = np.argmax(result.detach().numpy())

    # plt.cla()
    # plt.bar(np.arange(0,90), result.detach().numpy())
    # fig.canvas.start_event_loop(0.0001)

    # First frame's prediction is equal to second frame's
    if i == 0:
        results.append(prediction)
    results.append(prediction)
    
    if i != 10796 and i % 20 == 0:
        print("{0}/10797    Predicted Speed: {1}    ".format(i, prediction), end='\r')
    elif i == 10796:
        print("{0}/10797    Predicted Speed: {1}    ".format(i, prediction))

with open('test_result.txt', 'w') as f:
    for i, item in enumerate(results):
        if i != len(results) - 1:
            f.write("%s\n" % item)
        else:
            f.write("%s" % item)