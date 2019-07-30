import numpy as np
import torch.nn as nn
import torchvision
import torch

class Speed_Classify_Model(nn.Module):
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
    model.train()
    loss = nn.BCELoss()
    transform = torchvision.transforms.ToTensor()
    frame_tensor = transform(frame)
    frame_tensor = torch.unsqueeze(frame_tensor, 0)
    optimizer.zero_grad()
    result = model(frame_tensor)

    target_tensor = np.zeros((90), dtype=np.float32)
    target_tensor[target] = 1.0
    target_tensor = torch.tensor(target_tensor)

    train_loss = loss(result, target_tensor)
    train_loss.backward()
    optimizer.step()

    return train_loss.sum().item(), result.detach().numpy()