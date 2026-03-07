import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image loader
loader = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device)

content = load_image("content.jpg")
style = load_image("style.jpg")

# use pretrained VGG19
model = models.vgg19(pretrained=True).features.to(device).eval()

generated = content.clone().requires_grad_(True)

optimizer = torch.optim.Adam([generated], lr=0.01)

for step in range(200):

    optimizer.zero_grad()

    gen_features = model(generated)
    content_features = model(content)
    style_features = model(style)

    content_loss = torch.mean((gen_features - content_features)**2)
    style_loss = torch.mean((gen_features - style_features)**2)

    loss = content_loss + 0.5 * style_loss

    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print("Step:", step, "Loss:", loss.item())

# save result
output = generated.squeeze().detach().cpu()
plt.imsave("stylized_output.png", output.permute(1,2,0))
print("Style transfer completed. Output saved as stylized_output.png")
