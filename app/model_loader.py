import sys
import os
sys.path.append(os.path.abspath("../module4-helper"))

import torch
from torchvision import transforms
from PIL import Image
from helper_lib.model import SimpleCNN
from helper_lib.utils import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
model = load_model(model, "../module4-helper/cnn.pth", device)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_image(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()
