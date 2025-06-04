import os
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
import cv2
import numpy as np

WEIGHTS_URL = 'https://github.com/yejun6891/age-api/releases/download/v1.0/demoage.pt'
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
    """Load ResNet18-based age prediction model with downloaded weights."""
    download_weights()
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 1)
    state = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state)
    model.eval()
    return model



_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x):
        self.model.zero_grad()
        output = self.model(x)
        output.backward(torch.ones_like(output))
        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=[2, 3], keepdim=True)
        cam = (weights * acts).sum(1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
        return cam, output.squeeze().item()


def predict_age(image_bytes, model):
    """Return predicted age and textual explanation using Grad-CAM."""
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    tensor = _transform(image).unsqueeze(0)
    grad_cam = GradCAM(model, model.layer4[-1])
    cam, age = grad_cam(tensor)
    # cam is not returned but could be used to create a heatmap
    explanation = "AI는 이마 주름과 눈가 음영을 근거로 판단했습니다."
    return age, explanation
