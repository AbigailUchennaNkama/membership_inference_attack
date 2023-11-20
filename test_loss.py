import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import resnet18
from alive_progress import alive_bar

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RNG = torch.Generator().manual_seed(42)


def setup_model():
    local_path = "weights_resnet18_cifar10.pth"
    weights_pretrained = torch.load(local_path, map_location=DEVICE)

    model = resnet18(weights=None, num_classes=10)
    model.load_state_dict(weights_pretrained)
    model.to(DEVICE)
    model.eval()
    return model

def compute_data_losses(net, loader):
    """Compute losses per sample"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        logits = net(inputs)
        losses = criterion(logits, targets).numpy(force=True)
        for l in losses:
            all_losses.append(l)

    return torch.tensor(all_losses)


def accuracy(net, loader):
    """Compute Accuracy on a dataset given by loader."""
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total



def main():

    print("Running on device:", DEVICE.upper())

    # download and pre-process CIFAR10
    normalize = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    )

    train_set = torchvision.datasets.CIFAR10(
       root="./data", train=True, download=True, transform=normalize
    )

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)

    # we split held out data into test and validation set
    held_out = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=normalize
    )

    test_set, val_set = torch.utils.data.random_split(held_out, [0.5, 0.5], generator=RNG)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2)

    #show_images()
    model = setup_model()
    model.eval()

    print(f"Train set accuracy: {100.0 * accuracy(model, train_loader):0.1f}%")
    print(f"Test set accuracy: {100.0 * accuracy(model, test_loader):0.1f}%")

    train_losses = compute_data_losses(model, train_loader)
    test_losses = compute_data_losses(model, test_loader)

# Plotting validation losses
    plt.plot(test_losses, '-o', label='Test Losses', color='blue')

# Plotting training losses
    plt.plot(train_losses, '-o', label='Training Losses', color='red')

# Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Test Loss per Sample')

# Show legend
    plt.legend()

# Display the plot
    plt.show()


if __name__  == "__main__":
    main()
