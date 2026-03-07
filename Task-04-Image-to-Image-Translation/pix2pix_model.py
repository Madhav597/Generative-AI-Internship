import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

# Simple Generator (Pix2Pix style)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Load image
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

image = Image.open("input.jpg")
image = transform(image).unsqueeze(0)

# Initialize generator
generator = Generator()

# Generate translated image
output = generator(image)

# Save result
save_image(output, "generated_output.png")

print("Image generated and saved as generated_output.png")
