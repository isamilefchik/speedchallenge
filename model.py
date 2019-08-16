import numpy as np
import torch.nn as nn
import torchvision
import torch

class Speed_Classify_Model(nn.Module):
    """ Largely based off of LeNet. """

    def __init__(self):
        super(Speed_Classify_Model, self).__init__()
        self.conv1 = nn.Conv2d(3,32,(5,5), stride=2)
        self.relu1 = nn.ReLU()
        self.max1 = nn.MaxPool2d((2,2),(2,2))

        self.conv2 = nn.Conv2d(32,16,(5,5), stride=2)
        self.relu2 = nn.ReLU()
        self.max2 = nn.MaxPool2d((2,2),(2,2))
        
        self.lin1 = nn.Linear(18096,90)
        self.soft = nn.Softmax(dim=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.max1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.max2(out)

        out = out.view(-1)

        out = self.lin1(out)
        out = self.soft(out)

        return out

def train_model(model, optimizer, frame, target):
    """ Model training routine. """

    model.train()
    loss = nn.BCELoss()
    transform = torchvision.transforms.ToTensor()
    frame_tensor = transform(frame)
    frame_tensor = torch.unsqueeze(frame_tensor, 0)
    optimizer.zero_grad()
    result = model(frame_tensor)

    target_tensor = make_gauss_target(target, 90, 1.0)

    train_loss = loss(result, target_tensor)
    train_loss.backward()
    optimizer.step()

    return train_loss.sum().item(), result.detach().numpy()

def make_gauss_target(target, n, sig):
    """ Makes normal distribution across classes according to ground truth speed. """

    gauss_target = np.zeros(n)
    for i in range(n):
        gauss_target[i] = np.exp(-np.power(i - target, 2.) / (2 * np.power(sig, 2.)))
    return torch.tensor(gauss_target, dtype=torch.float)

