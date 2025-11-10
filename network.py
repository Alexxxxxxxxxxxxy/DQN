import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_states, num_actions):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(num_states, 32 ,kernel_size=8, stride=4,bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2,bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1,bias=False)

        self.conv = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.conv3,
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, num_actions)

        self.fc = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2
        )

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        x = self.fc(x)
        return x