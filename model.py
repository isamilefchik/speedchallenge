import torch.nn as nn

class Speed_Classify_Model(nn.Module):
    def __init__(self):
        super(Speed_Classify_Model, self).__init__()
        self.pad1 = nn.ReflectionPad2d((7,7,7,7))
        self.conv1 = nn.Conv2d(3, 64, (15,15))
        self.subsamp = nn.MaxPool2d((5,5))


    def forward(self, x):
        out = self.pad1(x)
        return out

def train_model(model, frame):

