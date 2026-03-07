import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.down1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.down2 = nn.Conv2d(64, 128, 4, 2, 1)

        self.up1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(64, 3, 4, 2, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):

        d1 = self.relu(self.down1(x))
        d2 = self.relu(self.down2(d1))

        u1 = self.relu(self.up1(d2))
        output = self.tanh(self.up2(u1))

        return output
