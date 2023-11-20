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
from torchvision.datasets.utils import download_url
import tarfile
from termcolor import colored
from torchvision.models import resnet18

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BinaryClassifier(nn.Module):
    def __init__(self, input_size=3):  # Adjust input_size to match the data size
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# Function to load a model
def load_target_model():
    local_path = "weights_resnet18_cifar10.pth"
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

def predict_membership(target_model_path, input_data_path, attack_model_path):
    # Load target model
    target_model = load_target_model()
    # Apply the same transformation used during training
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

    # Set the attack model to evaluation mode
    attack_model.eval()

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
def membership_inference():
    #download and pre-process CIFAR10
    print("Creating sample dataset...")
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    download_url(dataset_url, '.')

    with tarfile.open("./cifar10.tgz", "r:gz") as tar:
        tar.extractall(path="./data")
    print('---'*45)

    # Example usage with a datapoint from the CIFAR-10 dataset
    target_model_path = "./weights_resnet18_cifar10.pth"
    attack_model_path = "./attack_model.pth"
    example_image_path = "/content/data/cifar10/test/cat/0004.png"

    # Make the prediction
    predict_membership(target_model_path, example_image_path, attack_model_path)

if __name__ == "__main__":
    membership_inference()
