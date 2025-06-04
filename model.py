import os
import requests
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO

WEIGHTS_URL = 'https://example.com/demoage.pt'
WEIGHTS_PATH = 'demoage.pt'


def download_weights():
    if not os.path.exists(WEIGHTS_PATH):
        resp = requests.get(WEIGHTS_URL, stream=True)
        resp.raise_for_status()
        with open(WEIGHTS_PATH, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def load_model():
    download_weights()
    model = torch.jit.load(WEIGHTS_PATH, map_location=torch.device('cpu'))
    model.eval()
    return model


_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_age(image_bytes, model):
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    tensor = _transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
    return output.squeeze().item()
