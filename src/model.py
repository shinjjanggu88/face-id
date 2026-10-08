import torch
import torch.nn as nn
class FaceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3) 
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32,3)
        self.fc1 = nn.Linear(32 * 30 * 30, 64)
        self.fc2 = nn.Linear(64, 4)
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 32 * 30 * 30)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x