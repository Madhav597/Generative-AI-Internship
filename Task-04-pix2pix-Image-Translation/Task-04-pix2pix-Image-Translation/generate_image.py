import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pix2pix_model import Generator

# image preprocessing
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

# load image
image = Image.open("input.jpg").convert("RGB")
image = transform(image).unsqueeze(0)

# load model
model = Generator()

# generate image
with torch.no_grad():
    output = model(image)

# save result
save_image(output, "output.png")

print("Image generated successfully")
