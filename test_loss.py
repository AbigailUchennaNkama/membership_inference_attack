import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import resnet18

from jsonargparse import CLI
from alive_progress import alive_bar


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RNG = torch.Generator().manual_seed(42)

class CIFAR10WithNoise(torchvision.datasets.CIFAR10):
    def __init__(self, noise_std=0.1, **kwargs):
        super().__init__(**kwargs)
        self.noise_std = noise_std

    def __getitem__(self, index):
        image, label = super().__getitem__(index)

        # Add Gaussian noise
        noise = torch.randn(image.size()) * self.noise_std
        noisy_image = image + noise
        # Clip the image to make sure the values are between 0 and 1
        noisy_image = torch.clamp(noisy_image, 0, 1)

        return noisy_image, label


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

def kl_divergence(p, q):
    """Compute KL Divergence"""
    return (p * (p / q).log()).sum()

def compute_kl_divergences(model, dataloader, threshold=0):
    """Compute KL Divergence for a DataLoader"""
    kl_divs = []
    for i, data in enumerate(dataloader, 0):
        inputs, _ = data
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        probabilities = nn.Softmax(dim=1)(outputs)
        uniform_dist = torch.full_like(probabilities, 1.0 / 10)

        for prob in probabilities:
            kl_div = kl_divergence(prob, uniform_dist)
            if kl_div < threshold:
                kl_divs.append(kl_div.item())
    return kl_divs

def main(threshold: int=120, gn: float=0.08):
    """Plot the KL Divergence of distribution.

    Args:
        threshold: threshold for KL divergence.
        gn: Gaussian Noise for distribution.
    """

    print("Running on device:", DEVICE.upper())
    print(threshold)
    print(gn)
    # download and pre-process CIFAR10
    normalize = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    )

    gnoise = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    )
    # Instantiate the dataset with noise added
    cifar10_test = CIFAR10WithNoise(
        root='./data',
        train=False,
        download=True,
        transform= normalize,
        noise_std=gn
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
    #Create a DataLoader for the test set
    gn_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=128, shuffle=False, num_workers=2)

    #show_images()
    model = setup_model()
    model.eval()

    #print(f"Train set accuracy: {100.0 * accuracy(model, train_loader):0.1f}%")
    #print(f"Test set accuracy: {100.0 * accuracy(model, test_loader):0.1f}%")

    #train_losses = compute_data_losses(model, train_loader)
    #test_losses = compute_data_losses(model, test_loader)
    train_kl_divergences = compute_kl_divergences(model, train_loader, threshold)
    gn_kl_divergences = compute_kl_divergences(model, gn_loader, threshold)
    test_kl_divergences = compute_kl_divergences(model, test_loader, threshold)


# Plot the KL divergences for train and test set
    plt.figure(figsize=(10, 6))
    plt.hist(gn_kl_divergences, bins=50, alpha=0.5, color='cyan', label='GN KL Divergences')
    plt.hist(test_kl_divergences, bins=50, alpha=0.5, color='yellow', label='Test KL Divergences')
    plt.hist(train_kl_divergences, bins=50, alpha=0.5, color='magenta', label='Train KL Divergences')
    plt.xlabel('KL Divergence')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.title(f'KL Divergence Distributions for CIFAR-10 Train and Test and GN={gn} Sets')
    plt.show()

if __name__  == "__main__":
    CLI(main)
