import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
from PIL import Image
from IPython.display import HTML
import os
import shutil
from pathlib import Path
import random
import tarfile
from termcolor import colored
from torchvision.models import resnet18
from torchvision.datasets.utils import download_url
from model_architecture import BinaryClassifier, load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Function to load a model
def load_target_model(target_model_path):
    local_path = target_model_path
    weights_pretrained = torch.load(local_path, map_location=DEVICE)

    model = resnet18(weights=None, num_classes=10)
    model.load_state_dict(weights_pretrained)
    model.to(DEVICE)
    model.eval()
    return model

def load_attack_model():
    local_path = './attack_model.pth'
    weights_pretrained = torch.load(local_path, map_location=DEVICE)

    # Initialize the model
    model = BinaryClassifier()
    model.load_state_dict(weights_pretrained)
    model.to(DEVICE)
    model.eval()
    return model

# Function to predict membership

def predict_membership(target_model_path, input_data_path):
    # Load target model
    target_model = load_target_model(target_model_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Preprocess input_data
    with torch.no_grad():
        input_data = Image.open(input_data_path)
        input_data = transform(input_data).unsqueeze(0).to(DEVICE)
        target_logits = target_model(input_data)
        target_predictions = torch.softmax(target_logits, dim=1)
        top_k_values, _ = torch.topk(target_predictions, k=3)

    # Load attack model
    attack_model = load_attack_model()

    # Forward pass through attack model
    with torch.no_grad():
        attack_logits = attack_model(top_k_values)
        attack_predictions = torch.softmax(attack_logits, dim=1)
        attack_prob = round(torch.max(attack_predictions, dim=1).values[0].item(), 2) * 100
        attack_label = attack_predictions.argmax().item()

    # Check attack model's prediction
    if attack_label == 1:
        result = f'[RESULT] Input is a Member at {attack_prob}% certainty'
        colored_result = f'<font color="green">{result}</font>'
    else:
        result = f'[RESULT] Input is not a Member at {attack_prob}% certainty'
        colored_result = f'<font color="red">{result}</font>'

    display(HTML(colored_result))
