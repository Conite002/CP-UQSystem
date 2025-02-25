import torch.nn as nn
import torch.nn.functional as F


class SimpleMNISTModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)      
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.num_classes = num_classes
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)          
        x = x.view(-1, 64 * 12 * 12)    
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
